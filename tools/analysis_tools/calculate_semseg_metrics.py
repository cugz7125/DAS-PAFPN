import pickle
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools import mask as mask_util
import json

def calculate_metrics(preds, gts):
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for pred_mask, gt_mask in zip(preds, gts):
        tp += np.sum(np.logical_and(pred_mask, gt_mask))
        fp += np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
        fn += np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))
        tn += np.sum(np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)))
        
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    oa = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return dict(IoU=iou, Precision=precision, Recall=recall, F1=f1, OA=oa)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_pkl', help='Path to prediction .pkl file')
    parser.add_argument('ann_file', help='Path to coco annotation .json file')
    args = parser.parse_args()

    print(f"Loading predictions from {args.prediction_pkl}...")
    preds_data = pickle.load(open(args.prediction_pkl, 'rb'))
    print(f"Loading annotations from {args.ann_file}...")
    coco_data = json.load(open(args.ann_file, 'r'))
    
    # 将 GT 标注按 image_id 组织起来
    gt_masks_by_imgid = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        h, w = ann['image_height'], ann['image_width']
        if img_id not in gt_masks_by_imgid:
            gt_masks_by_imgid[img_id] = np.zeros((h, w), dtype=bool)
        
        rle = mask_util.frPyObjects(ann['segmentation'], h, w)
        mask = mask_util.decode(rle)
        if len(mask.shape) > 2: mask = np.sum(mask, axis=2)
        gt_masks_by_imgid[img_id] = np.logical_or(gt_masks_by_imgid[img_id], mask > 0)

    all_pred_sem_masks = []
    all_gt_sem_masks = []
    
    print("Converting instance masks to semantic masks...")
    for pred in tqdm(preds_data):
        img_id = pred.img_id
        h, w = pred.height, pred.width
        
        if img_id not in gt_masks_by_imgid:
            continue

        # 合并预测的实例掩码
        pred_instances = pred.pred_instances
        if len(pred_instances) > 0:
            pred_sem_mask = np.any(pred_instances.masks.to_ndarray(), axis=0)
        else:
            pred_sem_mask = np.zeros((h, w), dtype=bool)
        
        all_pred_sem_masks.append(pred_sem_mask.flatten())
        all_gt_sem_masks.append(gt_masks_by_imgid[img_id].flatten())

    print("Calculating metrics...")
    all_preds_flat = np.concatenate(all_pred_sem_masks)
    all_gts_flat = np.concatenate(all_gt_sem_masks)
    
    metrics = calculate_metrics(all_preds_flat, all_gts_flat)
    
    print("\n--- Semantic Segmentation Metrics ---")
    for key, value in metrics.items():
        print(f"  - {key}: {value * 100:.2f}%")
    print("-----------------------------------")


if __name__ == '__main__':
    main()