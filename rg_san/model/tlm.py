import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean, scatter
# modified torch multihead attention
from ..torch.nn import MultiheadAttention
from .attention import MultiheadAttention as MultiheadAttention1
# graph
from .graph.graph_transformer_net import GraphTransformerNet
from .graph.layers.graph_transformer_edge_layer import GraphTransformerLayer, GraphTransformerSubLayer

from .transformer import TransformerDecoderLayer, MLP

from .position_embedding import PositionEmbeddingCoordsSine

import functools

class DDI(nn.Module):

    def __init__(
            self,
            hidden_dim,
            out_dim,
            n_heads,
            dropout=0.0,
            layer_norm=True, 
            batch_norm=False, 
            residual=True, 
            use_bias=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.graph_attn = GraphTransformerSubLayer(hidden_dim, out_dim, n_heads, dropout, layer_norm, batch_norm)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def graph2batch(self, batched_graph):
        node_num = batched_graph.batch_num_nodes()
        batch_offsets = torch.cat([torch.tensor((0,), dtype=torch.int).to(batched_graph.device), node_num.cumsum(0).int()], dim=0)
        batch_data, batch_masks = self.get_batches(batched_graph.ndata['h'], batch_offsets)
        return batch_data, batch_masks
    
    def batch2graph(self, batch_data, batch_masks):
        B = batch_data.shape[0]
        batch_x = []
        for i in range(B):
            batch_x.append(batch_data[i, ((~batch_masks)[i])])
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x
    
    def forward(self, x, x_mask, batch_g, batch_x, batch_e, pe=None, cat='parallel'):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        # parallel
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        sa_output, _ = self.self_attn(q, k, x, key_padding_mask=x_mask)

        # graph-attention
        batch_x, batch_e = self.graph_attn(batch_g, batch_x, batch_e)
        batch_g.ndata['h'] = batch_x
        batch_g.edata['e'] = batch_e

        # transform batched graph to batched tensor
        ga_output, _ = self.graph2batch(batch_g)
        ga_output = torch.cat([ga_output, torch.zeros(B, x_mask.shape[1]-ga_output.shape[1], ga_output.shape[-1]).to(ga_output.device)], dim=1)

        # residual connection
        output = self.dropout(sa_output + ga_output) + x
        output = self.norm(output)

        return output, batch_e

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
    
class TLM(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        kernel='target_id',
        query_init='cross_attn',
        global_feat='mean',
        attn_mask_thresh = 0.1,
        temperature=10000,
        pos_type='fourier',
        graph_params=None,
        decoder=None,
        bidirectional=False,
        cls_root=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.attn_mask_thresh = attn_mask_thresh
        self.d_model = d_model
        self.nhead = nhead
        self.cls_root = cls_root
        H = 768
        self.lang_proj = nn.Linear(H, 256)
        
        if query_init=='cross_attn':
            self.init_ca_query = nn.Linear(d_model, d_model)
            self.init_ca_key = nn.Linear(d_model, d_model)
            self.init_ca_value = nn.Linear(d_model, d_model)
            self.init_ca_norm = nn.LayerNorm(d_model)
            self.cross_attn = MultiheadAttention1(d_model, nhead, dropout=dropout, vdim=d_model)
        else:
            raise NotImplementedError
        
        self.decoder_norm = nn.LayerNorm(d_model)

        self.lap_pos_enc = graph_params['lap_pos_enc']
        if graph_params['lap_pos_enc']:
            pos_enc_dim = graph_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_model)

        self.embedding_root = nn.Linear(1, d_model)
        if bidirectional:
            self.embedding_e = nn.Embedding(2*graph_params['num_bond_type'], d_model)
        else:
            self.embedding_e = nn.Embedding(graph_params['num_bond_type'], d_model)
            
        self.ddi_layers = nn.ModuleList([])
        self.ddi_ffn_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.ddi_layers.append(DDI(graph_params['hidden_dim'], graph_params['out_dim'], graph_params['n_heads'], graph_params['dropout'], graph_params['layer_norm'], graph_params['batch_norm'], graph_params['residual']))
            self.ddi_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.decoder_layers.append(TransformerDecoderLayer(**decoder))
        
        self.position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=False, pos_type=pos_type, d_pos=d_model)
        self.key_position_embedding = PositionEmbeddingCoordsSine(temperature=temperature, normalize=True, pos_type=pos_type, d_pos=d_model)
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self.offset = MLP(d_model, d_model, 3, 3)
        nn.init.constant_(self.offset.layers[-1].weight.data, 0)
        nn.init.constant_(self.offset.layers[-1].bias.data, 0)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.query_init = query_init
        self.global_feat = global_feat
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, batch_mask, sp_mask_features): 
        pred_masks = torch.einsum('bnd,bmd->bnm', query, sp_mask_features)
        
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < self.attn_mask_thresh).bool() # [B, 1, num_sp]    
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None

        return pred_masks, attn_masks

    def prediction_head(self, query, batch_mask, sp_mask_features):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, batch_mask, sp_mask_features)
        return pred_scores, pred_masks, attn_masks

    def graph2batch(self, batched_graph):
        node_num = batched_graph.batch_num_nodes()
        batch_offsets = torch.cat([torch.tensor((0,), dtype=torch.int).to(batched_graph.device), node_num.cumsum(0).int()], dim=0)
        batch_data, batch_masks = self.get_batches(batched_graph.ndata['h'], batch_offsets)
        return batch_data, batch_masks
    
    def batch2graph(self, batch_data, batch_masks):
        B = batch_data.shape[0]
        batch_x = []
        for i in range(B):
            batch_x.append(batch_data[i, ((~batch_masks)[i])])
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x
    
    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = (~lang_masks).sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * (~lang_masks).unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len
    
    def before_process(self, sp_coords_float, batch_offsets):
        B = len(batch_offsets)-1
        input_ranges = []
        lengths = batch_offsets[1:] - batch_offsets[:-1]
        max_length = lengths.max().item()
        pos_batched = sp_coords_float.new_zeros(max_length, B, self.d_model)
        for i in range(B):
            start, end = batch_offsets[i], batch_offsets[i+1]
            pos_i = sp_coords_float[start:end]
            pos_i_min, pos_i_max = pos_i.min(0)[0], pos_i.max(0)[0]
            pos_emb_i = self.key_position_embedding(pos_i.unsqueeze(0), num_channels=self.d_model, input_range=(pos_i_min.unsqueeze(0), pos_i_max.unsqueeze(0)))[0] 
            pos_batched[:lengths[i], i, :] = pos_emb_i
            input_ranges.append((pos_i_min, pos_i_max))
        input_ranges_mins, input_ranges_maxs = [], []
        for i in range(len(input_ranges)):
            pos_i_min, pos_i_max = input_ranges[i]
            input_ranges_mins.append(pos_i_min) #[3]
            input_ranges_maxs.append(pos_i_max)
        input_ranges_mins = torch.stack(input_ranges_mins, dim=0).unsqueeze(0) #[1, bsz, 3]
        input_ranges_maxs = torch.stack(input_ranges_maxs, dim=0).unsqueeze(0) #[1, bsz, 3]
        
        return input_ranges, pos_batched, input_ranges_mins, input_ranges_maxs 
    
    def forward_iter_pred(self, x, batch_offsets, batched_graph, lang_feats=None, lang_masks=None, sp_coords_float=None, nsubj_inds=None):
        """
        x [B*M, inchannel]
        """
        # process the graph feats
        B = len(batch_offsets) - 1
        d_model = self.d_model
        
        batched_graph = batched_graph.to(x.device)
        try:
            batch_lap_pos_enc = batched_graph.ndata['lap_pos_enc'].to(x.device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(x.device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        # prepare for DDI
        lang_feats = self.lang_proj(lang_feats) 
        cls_feat = lang_feats[:, 0, :].unsqueeze(1)
        lang_feats = lang_feats[:, 1:, :] 
        lang_len = lang_masks.sum(-1) - 2 
        lang_masks = torch.arange(lang_feats.shape[1])[None, :].to(lang_feats.device) < lang_len[:, None]  
        lang_masks = ~lang_masks 
        query = lang_feats

        if self.cls_root:
            # cls_feat as root node
            batch_x = torch.cat([torch.cat([cls_feat[i], query[i][~(lang_masks[i].bool())]], dim=0) for i in range(query.shape[0])], dim=0)
        else:
            root_embedding = self.embedding_root(torch.tensor((0,)).float().to(x.device)).unsqueeze(0)
            # every sentence has a root node, there are B sentences
            assert (lang_len+1 - batched_graph.batch_num_nodes()).sum() == 0
            batch_x = torch.cat([torch.cat([root_embedding, query[i][~(lang_masks[i].bool())]], dim=0) for i in range(query.shape[0])], dim=0)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(batch_lap_pos_enc.float()) 
            batch_x = batch_x + h_lap_pos_enc
        batched_graph.ndata['h'] = batch_x
        
        # add root node
        query, _ = self.graph2batch(batched_graph)
        query = torch.cat([query, torch.zeros(lang_feats.shape[0], lang_feats.shape[1]-query.shape[1], query.shape[-1]).to(query.device)], dim=1)
        lang_masks = torch.arange(lang_feats.shape[1])[None, :].to(lang_feats.device) <= lang_len[:, None] 
        lang_masks = ~lang_masks 

        batch_e = batched_graph.edata['feat'].to(x.device)
        batch_e = self.embedding_e(batch_e)
        batched_graph.edata['e'] = batch_e

        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)

        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        coords_float_batched, _ = self.get_batches(sp_coords_float, batch_offsets)
        prediction_masks = []
        prediction_scores = []

        input_ranges, pos_batched, input_ranges_mins, input_ranges_maxs = self.before_process(sp_coords_float, batch_offsets)

        memory_key_padding_mask = batch_mask
        
        memory_mask = None
        prediction_centers = []
        
        # multi-round
        for i in range(self.num_layer):
            # DDI
            if not self.cls_root:
                batch_x = self.batch2graph(query, lang_masks)
                query, batch_e = self.ddi_layers[i](query, lang_masks, batched_graph, batch_x, batch_e)
                query = self.ddi_ffn_layers[i](query)

            if i==0 and self.query_init=='cross_attn':
                
                q = self.init_ca_query(query).transpose(0, 1)
                k = self.init_ca_key(inst_feats).transpose(0, 1)
                v = self.init_ca_value(inst_feats).transpose(0, 1)
                
                #out(N,B,d)
                out, weight, _ = self.cross_attn(q,k,v,key_padding_mask=memory_key_padding_mask)
                query = query.transpose(0, 1) + out
                query = self.init_ca_norm(query)#content query
                pquery = torch.einsum('bnm,bmd->bnd', weight, coords_float_batched) 

                reference_points = ((pquery.transpose(0,1) - input_ranges_mins)/(input_ranges_maxs - input_ranges_mins)).transpose(0,1)#position query 归一化尺寸
                query = query.permute(1,0,2).contiguous()
                
                if self.kernel=='target_id':
                    ind = nsubj_inds.unsqueeze(-1).unsqueeze(-1)
                    prediction_centers.append(pquery.gather(dim=1, index=ind.repeat(1, 1, pquery.size(-1))).squeeze(1))
                    pred_scores, pred_masks, _ = self.prediction_head(query.gather(dim=1, index=ind.repeat(1, 1, query.size(-1))), batch_mask, mask_feats)
                    _, _, attn_masks = self.prediction_head(query, batch_mask, mask_feats)
                    prediction_scores.append(pred_scores)
                    prediction_masks.append(pred_masks)
                else: raise NotImplementedError
            
            obj_center = reference_points[..., :3].transpose(0, 1).contiguous()  # [num_queries, batch_size, 3]

            reference_points_coords_float = torch.zeros_like(reference_points) #[batch_size, num_queries, 3] 
    
            for b in range(B):
                pos_i_min, pos_i_max = input_ranges[b] #[3]
                reference_points_coords_float[b] = reference_points[b] * (pos_i_max - pos_i_min) + pos_i_min
            reference_points_coords_float = reference_points_coords_float.transpose(0, 1) #[num_queries, batch_size, 3]

            query_sine_embed = self.position_embedding(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            memory_key_padding_mask = batch_mask
            
            #(B,N,d), (B,N,M)
            query, src_weight = self.decoder_layers[i](query,inst_feats,query_coords_float=reference_points_coords_float,
                                           key_coords_float=coords_float_batched,tgt_key_padding_mask=lang_masks,
                                           memory_mask=memory_mask,memory_key_padding_mask=memory_key_padding_mask,pos=pos_batched,
                                           query_pos=query_pos,query_sine_embed=query_sine_embed)
            
            
            query_norm = self.decoder_norm(query.transpose(0,1))
            obj_center_offset = self.offset(query_norm) #[num_queries, bsz, 3] 
            new_reference_points_y = obj_center * (input_ranges_maxs - input_ranges_mins) + input_ranges_mins + obj_center_offset #[num_queries, bsz, 3]
            new_reference_points = (new_reference_points_y - input_ranges_mins) / (input_ranges_maxs - input_ranges_mins) #[num_queries, bsz, 3]
            new_reference_points = new_reference_points.transpose(0,1) #[bsz, num_queries, 3]
            reference_points = new_reference_points.detach()
            
            if self.kernel=='target_id':
                ind = nsubj_inds.unsqueeze(-1).unsqueeze(-1)
                prediction_centers.append(new_reference_points_y.transpose(0,1).gather(dim=1, index=ind.repeat(1, 1, new_reference_points_y.size(-1))).squeeze(1))        
                pred_scores, pred_masks, _ = self.prediction_head(query.gather(dim=1, index=ind.repeat(1, 1, query.size(-1))), batch_mask, mask_feats)
            else: raise NotImplementedError
            
            _, _, attn_masks = self.prediction_head(query, batch_mask, mask_feats)
            
            memory_mask = attn_masks.unsqueeze(1).expand(-1, self.nhead, -1, -1).contiguous().flatten(0,1)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        
        return {
            'masks':
            pred_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'ref_center':
            prediction_centers[-1],
            'aux_outputs': [{
                'masks': a,
                'scores': b,
                'center': c,
            } for a, b, c in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
                prediction_centers[:-1],
            )],
        }

    def forward(self, x, batch_offsets, batched_graph, lang_feats=None, lang_masks=None, sp_coords_float=None, nsubj_inds=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets, batched_graph, lang_feats, lang_masks, sp_coords_float, nsubj_inds)
        else:
            raise NotImplementedError
