import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Union

@torch.jit.script
def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob)**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum('nc,mc->nm', focal_pos, targets) + torch.einsum('nc,mc->nm', focal_neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  
    return loss


def get_iou(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def get_iou_prob(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  
    return loss.mean()

@torch.jit.script
def dice_loss_prob(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  
    return loss.mean()

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

@torch.jit.script
def dice_loss_multi_calsses(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float] = None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from
    https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1.0 - per_channel_dice

    return loss.mean()

@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        loss_weight=[1.0, 1.0, 0.5, 0.5],
        loss_fun='bce'
    ):
        super().__init__()
        self.loss_fun = loss_fun
        self.L1 = nn.L1Loss(reduction='none')
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.ce = nn.CrossEntropyLoss()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, layer, aux_outputs, pad_masks, gt_spmasks, gt_centers):
        loss_out = {}

        pred_scores = aux_outputs['scores'].squeeze()
        pred_masks = aux_outputs['masks'].squeeze()
        if len(pred_masks.shape)==1: pred_masks = pred_masks.unsqueeze(0)
        pred_center = aux_outputs['center'].squeeze()
        if len(pred_center.shape)==1: pred_center = pred_center.unsqueeze(0)
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        
        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # mask loss
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)
        
        
        gt_centers_mask = gt_centers>-999
        center_loss = torch.masked_select(self.L1(gt_centers,pred_center), gt_centers_mask).mean()

        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['center_loss'] = center_loss

        loss = (self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss + self.loss_weight[2] * score_loss + self.loss_weight[3] * center_loss)

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, gt_spmasks, gt_centers):
        
        loss_out = {}

        pred_scores = pred['scores'].squeeze()
        pred_masks = pred['masks'].squeeze()
        if len(pred_masks.shape)==1: pred_masks = pred_masks.unsqueeze(0)
            
        pad_masks = (~(pred['batch_mask'])).squeeze()
        if len(pad_masks.shape)==1: pad_masks = pad_masks.unsqueeze(0)
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        
        if len(pred['ref_center'].shape)==1: pred['ref_center'] = pred['ref_center'].unsqueeze(0)
        
        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # mask loss
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()
            
        # dice loss
        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)
        
        #center_loss
        gt_centers_mask = gt_centers>-999
        center_loss = torch.masked_select(self.L1(gt_centers,pred['ref_center']), gt_centers_mask).mean()
        
        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['center_loss'] = center_loss
        loss = (self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss + self.loss_weight[2] * score_loss + self.loss_weight[3] * center_loss)
        
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, gt_spmasks, gt_centers)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss

        return loss, loss_out