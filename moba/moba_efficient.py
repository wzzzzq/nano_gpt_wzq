"""A clean version of efficient moba implementation with flash-attn"""

import torch

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn
        _, _, _, _, self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn
        _, _, _, _, moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv, _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    fixed_chunk_ratio: float = 0,  # Number of fixed chunks to use (evenly spaced)
) -> torch.Tensor:
    """An efficient version of moba implementation with evenly-spaced fixed chunks
    
    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor
        max_seqlen (int): the max sequence length of the batch
        moba_chunk_size (int): size of chunks for MOBA attention
        moba_topk (int): number of dynamic chunks to select
        fixed_chunk_ratio (double): ratio ([0,1]) of fixed chunks to always use (evenly spaced)
    
    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    kv = torch.stack((k, v), dim=1)

    # Basic variables
    seqlen, num_head, head_dim = q.shape

    # Prepare chunk meta
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # Handle fixed chunks
    fixed_chunk_count = num_filtered_chunk * fixed_chunk_ratio
    dynamic_topk = moba_topk - fixed_chunk_count
    need_moba_attn = (dynamic_topk > 0) or (fixed_chunk_count > 0)

    # Corner case: if no MOBA attention needed, just return self attention
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # Create filtered_kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    # Calculate key_gate_weight and gate
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)
    key_gate_weight = key_gate_weight.type(torch.float32)
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [F_N_CHUNK, HEAD, SEQ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # Process gate, masking unchosen batch and apply causal mask
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    # Create fixed chunk mask - evenly spaced from first to last
    fixed_chunk_mask = torch.zeros(num_filtered_chunk, dtype=torch.bool, device=q.device)
    if fixed_chunk_count > 0:
        if fixed_chunk_count == 1:
            fixed_chunk_indices = torch.tensor([0], device=q.device)
        else:
            # Create evenly spaced indices from 0 to num_filtered_chunk-1
            fixed_chunk_indices = torch.linspace(
                0, num_filtered_chunk-1, fixed_chunk_count, 
                dtype=torch.long, device=q.device
            )
        fixed_chunk_mask[fixed_chunk_indices] = True
    
    gate_mask = torch.zeros_like(gate, dtype=torch.bool)
    
    # Always include fixed chunks in the gate mask
    for head_idx in range(num_head):
        gate_mask[fixed_chunk_mask, head_idx, :] = ~gate.isinf()[fixed_chunk_mask, head_idx, :] 

    # For remaining slots, select dynamic chunks
    dynamic_gate = gate.clone()
    if dynamic_topk > 0:
        # Mask out fixed chunks (already selected) from dynamic selection
        for i in range(num_filtered_chunk):
            if fixed_chunk_mask[i]:
                dynamic_gate[i] = float('-inf')
                
        # Get dynamic top-k chunks
        _, dyn_top_k_idx = torch.topk(
            dynamic_gate, k=dynamic_topk, dim=0, largest=True, sorted=False
        )
        
        # Create and apply dynamic gate mask
        dyn_gate_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
        dyn_gate_mask = dyn_gate_mask.scatter_(dim=0, index=dyn_top_k_idx, value=True)
        gate_mask = torch.logical_or(gate_mask, dyn_gate_mask)

    # Combine with original causal mask to ensure only valid positions are attended to
    gate_mask = torch.logical_and(gate_mask, torch.logical_not(gate.isinf()))

    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [HS indices] * N
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    moba_q = rearrange(q, "s h d -> (h s) d").index_select(
        0, moba_q_indices
    )  # [selected_S, D]
    moba_q = moba_q.unsqueeze(1)
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    # Prepare moba kv
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        moba_kv = moba_kv[valid_expert_mask]
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Wrap up the rest
    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
    )
