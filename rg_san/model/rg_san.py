import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools

from rg_san.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou
from .tlm import TLM

from transformers import MPNetModel

class RGSAN(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        tlm=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.decoder_param = tlm

        self.text_encoder = MPNetModel.from_pretrained('./backbones/mpnet-base')

        # tlm
        self.tlm = TLM(**tlm, in_channel=media)
        # criterion
        self.criterion = Criterion(**criterion)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(RGSAN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, batched_graph, lang_tokenss, lang_masks, coords_float, gt_centers, nsubj_inds, scenes_len=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, batch_offsets = self.expand(sp_feats, sp_coords_float, batch_offsets, scenes_len)
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        out = self.tlm(sp_feats, batch_offsets, batched_graph,  lang_feats, lang_masks, sp_coords_float, nsubj_inds)
        
        loss, loss_dict = self.criterion(out, gt_spmasks, gt_centers)
        
        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, batched_graph, lang_tokenss, lang_masks, coords_float, gt_centers, nsubj_inds, scenes_len=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)
        
        sp_feats, sp_coords_float, batch_offsets = self.expand(sp_feats, sp_coords_float, batch_offsets, scenes_len)
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.tlm(sp_feats, batch_offsets, batched_graph, lang_feats, lang_masks, sp_coords_float, nsubj_inds)
            
        ret = self.predict_by_feat(scan_ids, object_ids, ann_ids, out, superpoints, gt_pmasks, gt_spmasks)
        
        return ret
      
    def predict_by_feat(self, scan_ids, object_ids, ann_ids, out, superpoints, gt_pmasks, gt_spmasks):
        
        spious, pious, pred_pmasks, scan_idss = [], [], [], []
        b = len(gt_pmasks)
        for i in range(b):
            gt_pmask = gt_pmasks[i]
            gt_spmask = gt_spmasks[i]
            pred_spmask = out['masks'][i].squeeze()

            spiou = get_iou(pred_spmask, gt_spmask)

            pred_pmask = pred_spmask[superpoints]
            piou = get_iou(pred_pmask, gt_pmask)
            
            spious.append(spiou.cpu())
            pious.append(piou.cpu())
            pred_pmasks.append(pred_pmask.sigmoid().cpu())
            scan_idss.append(scan_ids[0])
        gt_pmasks = [gt_pmask.cpu() for gt_pmask in gt_pmasks]
        return dict(scan_id=scan_idss, object_id=object_ids, ann_id=ann_ids, piou=pious, spiou=spious, gt_pmask=gt_pmasks, pred_pmask=pred_pmasks)

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    
    def expand(self, sp_feats, sp_coords_float, batch_offsets, scenes_len):
        if scenes_len==None: return sp_feats, sp_coords_float, batch_offsets
        
        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                
        return sp_feats_expand, sp_coords_float_expand, batch_offsets_expand