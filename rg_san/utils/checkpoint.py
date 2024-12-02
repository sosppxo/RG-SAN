import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import json
from .mask_encoder import rle_decode, rle_encode


def save_single_instance(root, scan_id, object_id, ann_id, pred_pmask, all_pred_pmask=None):
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    # np.savetxt(osp.join(root,'predicted_masks', f'{scan_id}_{object_id}_{ann_id}.txt'), pred_pmask, fmt='%f')
    # save the encoded mask (json file)
    # pred_pmask need to be binarized
    # mask_encoded = rle_encode((pred_pmask>0.5).int().numpy())
    with open(osp.join(root,'predicted_masks', f'{scan_id}_{object_id}_{ann_id}.json'), 'w') as f:
        print(f'{scan_id}_{object_id}_{ann_id}.json')
        json.dump(pred_pmask, f)
    
    if all_pred_pmask is not None:
        with open(osp.join(root,'predicted_masks', f'{scan_id}_{object_id}_{ann_id}_all_pred_masks.json'), 'w') as f:
            # print(f'{scan_id}_{object_id}_{ann_id}.json')
            json.dump(all_pred_pmask, f)
    
    # with open(osp.join(root,'predicted_masks', f'{scan_id}_{object_id}_{ann_id}.json'), 'w') as f:
    #     json.dump(pred_pmask, f)


def save_pred_instances(root, name, scan_ids, object_ids, ann_ids, pred_pmasks, all_pred_pmasks=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    # print("encoding...")
    # need to be binarized
    # pred_pmasks = [rle_encode((pred_pmask>0.5).int().numpy()) for pred_pmask in pred_pmasks]
    # print("encoding done, saving...")
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, object_ids, ann_ids, pred_pmasks, all_pred_pmasks))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()
