import argparse
import gorilla, os
import torch
from tqdm import tqdm
import numpy as np
from rg_san.dataset import build_dataloader, build_dataset
from rg_san.model import RGSAN
from rg_san.utils.mask_encoder import rle_decode, rle_encode
from rg_san.utils import get_root_logger, save_pred_instances
import json

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_id', type=int, default=[0], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_id, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger(log_file=args.checkpoint.replace('.pth', '.log'))

    model = RGSAN(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.val, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks, lang_words, all_masks, attn_map = [], [], [], [], [], [], [], [], [], []
    iou_dict = {}
    words_dict = {}
    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            res = model(batch, mode='predict')
            scan_ids.extend(res['scan_id'])
            object_ids.extend(res['object_id'])
            ann_ids.extend(res['ann_id'])
            pious.extend(res['piou'])
            spious.extend(res['spiou'])
            gt_pmasks.extend(res['gt_pmask'])
            attn_map.extend([map.numpy() for map in res['pred_pmask']])
            
            pred_pmasks.extend(
                [
                    rle_encode((pred_pmask>0.5).int().numpy())
                    for pred_pmask in res['pred_pmask']
                ]
            )
            if 'lang_words' in batch:
                lang_words.extend(batch['lang_words'])
            if 'all_masks' in res:
                for j in range(len(res['all_masks'])):
                    all_masks.append(
                        [
                            rle_encode((pred_pmask>0.5).int().numpy())
                            for pred_pmask in res['all_masks'][j][1:len(batch['lang_words'][j])+1]
                        ]
                    )
            
            progress_bar.update()
            
        progress_bar.close()

    for idx, scan_id in enumerate(scan_ids):
        object_id = object_ids[idx]
        ann_id = ann_ids[idx]
        piou = pious[idx]
        iou_dict[scan_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)] = piou.item()
        if len(lang_words) > 0:
            words_dict[scan_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)] = lang_words[idx]
    iou_path = os.path.join(os.path.dirname(args.checkpoint), 'ious.json')
    # write to json
    with open(iou_path, 'w') as f:
        json.dump(iou_dict, f)
    
    if len(words_dict) > 0:
        words_path = os.path.join(os.path.dirname(args.checkpoint), 'words.json')
        with open(words_path, 'w') as f:
            json.dump(words_dict, f)
    else:
        all_masks = None
    
    for index in range(len(ann_ids)):
        k = scan_ids[index] + '_' + str(object_ids[index]).zfill(3) + '_' + str(ann_ids[index]).zfill(3)
        #if k in ['scene0030_00_052_002', 'scene0164_00_013_003', 'scene0621_00_028_000', 'scene0648_00_026_002']:
        if k in ['scene0011_00_018_002', 'scene0025_00_011_002', 'scene0046_00_001_001', 'scene0050_00_003_002', 'scene0187_00_005_004', 'scene0329_00_009_003', 'scene0356_00_014_002',
                 'scene0378_00_039_003', 'scene0378_00_043_004', 'scene0389_00_029_001', 'scene0426_00_008_003']:
            np.save(k+'.npy', attn_map[index])
            print(k, iou_dict[k])
    
    logger.info('Evaluate referring segmentation')
    # point-level metrics
    pious = torch.stack(pious, axis=0).cpu().numpy()
    # superpoint-level metrics
    spious = torch.stack(spious, axis=0).cpu().numpy()
    spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    spmiou = spious.mean()
    logger.info('sp_Acc@25: {:.3f}. sp_Acc@50: {:.3f}. sp_mIOU: {:.3f}.'.format(spprecision_quarter, spprecision_half, spmiou))
    
    print(pious.size)

    with open(os.path.join(cfg.data.val.data_root,"lookup.json"),'r') as load_f:
        # unique为1, multi为0
        unique_multi_lookup = json.load(load_f)
    unique, multi = [], []
    for idx, scan_id in enumerate(scan_ids):
        if unique_multi_lookup[scan_id][str(object_ids[idx])][str(ann_ids[idx])] == 0:
            unique.append(pious[idx])
        else:
            multi.append(pious[idx])
    unique = np.array(unique)
    multi = np.array(multi)
    for u in [0.25, 0.5]:
        logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
    logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
    
    # save output
    if args.out is None:
        output = input('If you want to save the results? (y/n)')
        if output == 'y':
            args.out = os.path.join(os.path.dirname(args.checkpoint), 'results')
        else:
            logger.info('Not saving results.')
            exit()
        
    if args.out:
        logger.info('Saving results...')
        save_pred_instances(args.out, 'pred_instance', scan_ids, object_ids, ann_ids, pred_pmasks, all_masks)
        logger.info('Done.')

if __name__ == '__main__':
    main()
