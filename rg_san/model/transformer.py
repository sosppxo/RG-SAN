# ------------------------------------------------------------------------
# Modified from Conditional DETR Transformer (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention
from .attention_rpe import MultiheadAttentionRPE
from .position_embedding import PositionEmbeddingCoordsSine

from timm.models.layers import DropPath, trunc_normal_
import einops

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, quant_grid_length, grid_size, rel_query=False, rel_key=False, rel_value=False, dim_feedforward=2048, dropout=0.1, activation="relu", abs_pos=True):
        super().__init__()
        
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        if abs_pos:
            self.cross_attn = MultiheadAttentionRPE(d_model*2, nhead, dropout=dropout, vdim=d_model)
        else:
            self.cross_attn = MultiheadAttentionRPE(d_model, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.quant_grid_length = quant_grid_length
        self.grid_size = grid_size
        self.rel_query, self.rel_key, self.rel_value = rel_query, rel_key, rel_value
        self.abs_pos = abs_pos
        
        if rel_query:
            self.relative_pos_query_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_query_table, std=.02)
        if rel_key:
            self.relative_pos_key_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_key_table, std=.02)
        if rel_value:
            self.relative_pos_value_table = nn.Parameter(torch.zeros(nhead, d_model//nhead, 3 * 2*quant_grid_length))
            trunc_normal_(self.relative_pos_value_table, std=.02)
    
    def forward_post(self, tgt, memory, query_coords_float, key_coords_float,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        key_coords_float = key_coords_float.transpose(0, 1)
                    
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        if self.abs_pos:
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        
        if self.abs_pos:
            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        else:
            q = q.view(num_queries, bs, n_model)
            k = k.view(hw, bs, n_model)
        
        # contextual relative position encoding
        # query_coords_float: [num_queries, B, 3]
        # key_coords_float: [max_length, B, 3]
        rel_pos = (query_coords_float.unsqueeze(1) - key_coords_float.unsqueeze(0)) #[num_queries, max_length, B, 3]
        
        rel_idx = torch.div(rel_pos, self.grid_size, rounding_mode='floor').long()
        rel_idx[rel_idx < -self.quant_grid_length] = -self.quant_grid_length
        rel_idx[rel_idx > self.quant_grid_length - 1] = self.quant_grid_length - 1
            
        rel_idx += self.quant_grid_length
        assert (rel_idx >= 0).all()
        assert (rel_idx <= 2*self.quant_grid_length-1).all()
            
        tgt2, _, src_weight= self.cross_attn(query=q,
                                    key=k,
                                    value=v, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask,
                                    rel_idx=rel_idx.contiguous(), 
                                    relative_pos_query_table=self.relative_pos_query_table if self.rel_query else None, 
                                    relative_pos_key_table=self.relative_pos_key_table if self.rel_key else None, 
                                    relative_pos_value_table=self.relative_pos_value_table if self.rel_value else None)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        tgt = tgt.transpose(0,1)
        return tgt, src_weight

    def forward(self, tgt, memory, query_coords_float, key_coords_float,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False,):
        return self.forward_post(tgt, memory, query_coords_float, key_coords_float, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")