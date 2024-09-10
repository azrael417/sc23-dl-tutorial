
from typing import List

import torch
import torch.distributed as dist
from utils import comm

def get_memory_format(tensor):
    """Helper routine to get the memory format"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

    
def sync_params(model):
    """Helper routine to ensure shared weights are the same after initialization"""
    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]

            for comm_group in param.is_shared_mp:
                if comm.get_size(comm_group) > 1:
                    tlist = [
                        torch.empty_like(param)
                        for x in range(comm.get_size(comm_group))
                    ]
                    tlist[comm.get_rank(comm_group)] = param
                    # gather all weights in the comm group
                    dist.all_gather(tlist, param, group=comm.get_group(comm_group))
                    # use weight of rank 0
                    # important to use copy here otherwise the handle gets detaches from the optimizer
                    param.copy_(tlist[0])

                    
# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    
    # treat trivial case first
    if num_chunks == 1:
        return [size]
    
    # first, check if we can split using div-up to balance the load: 
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks-1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


# distributed primitives
def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float().contiguous()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        input_ = input_.contiguous()
        dist.all_reduce(input_, group=group)

    return input_


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] >= num_chunks), f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into \
                                              {num_chunks} chunks. Empty slices are currently not supported."
    
    # get split
    sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    tensor_list = torch.split(tensor, sections, dim=dim)
    
    return tensor_list


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_
    
    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    
    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank]
    
    return output


def _gather(input_, dim_, shapes_, group=None):
    """Gather unevenly split tensors across ranks"""
    
    comm_size = dist.get_world_size(group=group)

    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError()
    if dim_ >= input_.dim():
        raise ValueError()

    if comm_size == 1:
        return input_

    # make contiguous:
    input_ = input_.contiguous()
    input_shape = list(input_.shape)

    if shapes_ is not None:
        input_list = []
        for src in range(comm_size):
            input_shape[dim_] = shapes_[src]
            input_list.append(torch.empty(input_shape, dtype=input_.dtype, device=input_.device))
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input_) for _ in range(comm_size)]

    dist.all_gather(input_list, input_, group=group)

    output = torch.cat(input_list, dim=dim_)

    return output
