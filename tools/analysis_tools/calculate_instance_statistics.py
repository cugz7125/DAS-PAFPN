# # tools/analysis_tools/calculate_instance_statistics.py (最终修复版)

# import argparse
# import pickle
# import json
# import os
# import numpy as np
# from tqdm import tqdm
# from pycocotools.coco import COCO
# from pycocotools import mask as mask_utils
# import torch

# def calculate_iou(mask1, mask2):
#     """
#     计算两个二进制 NumPy 掩码的 IoU (Intersection over Union)。
#     """
#     mask1 = mask1.astype(bool)
#     mask2 = mask2.astype(bool)
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     if union == 0:
#         return 0.0
#     return intersection / union

# def main():
#     parser = argparse.ArgumentParser(description='Calculate and compare instance statistics (count and area).')
#     parser.add_argument('prediction_file', help='Path to the prediction file (.pkl or .json).')
#     parser.add_argument('gt_file', help='Path to the ground-truth COCO annotation file (.json).')
#     parser.add_argument('--score-thresh', type=float, default=0.3, help='Confidence score threshold for predictions.')
#     args = parser.parse_args()

#     # --- 1. 加载真实标签 (GT) 并计算统计数据 ---
#     print(f"Loading ground truth from: {args.gt_file}")
#     try:
#         coco_gt = COCO(args.gt_file)
#     except Exception as e:
#         print(f"Error loading GT file: {e}")
#         return
    
#     gt_ann_ids = coco_gt.getAnnIds()
#     gt_anns = coco_gt.loadAnns(gt_ann_ids)
    
#     # 【关键修复】确保所有 GT 相关的统计变量在这里被正确定义
#     total_gt_instances = len(gt_anns)
    
#     if total_gt_instances == 0:
#         print("Error: No ground truth instances found in the annotation file.")
#         return
        
#     print(f"Found {total_gt_instances} ground truth instances.")
#     total_gt_area = sum([ann.get('area', 0) for ann in gt_anns])
#     avg_gt_area = total_gt_area / total_gt_instances

#     # --- 2. 加载预测结果并计算统计数据 ---
#     # print(f"Loading predictions from: {args.prediction_file}")
#     # file_ext = os.path.splitext(args.prediction_file)[1]
    
#     # total_pred_count = 0
#     # total_pred_area = 0.0

#     # if file_ext == '.pkl':
#     #     try:
#     #         from mmdet.structures import DetDataSample
#     #     except ImportError:
#     #         print("Error: Failed to import DetDataSample. Is MMDetection environment activated?")
#     #         return
            
#     #     pred_results = pickle.load(open(args.prediction_file, 'rb'))
#     #     print(f"Parsing {len(pred_results)} predictions from .pkl file...")
        
#     #     filtered_preds = []
#     #     for data_sample in pred_results:
#     #         if hasattr(data_sample, 'pred_instances'):
#     #             pred_instances = data_sample.pred_instances
#     #             if hasattr(pred_instances, 'scores'):
#     #                 keep = pred_instances.scores > args.score_thresh
#     #                 filtered_preds.append(pred_instances[keep])

#     #     total_pred_count = sum(len(p) for p in filtered_preds)
        
#     #     for pred_instances in tqdm(filtered_preds, desc="Calculating pred areas from PKL"):
#     #         if len(pred_instances) > 0 and hasattr(pred_instances, 'masks'):
#     #             masks_np = pred_instances.masks.to_ndarray()
#     #             total_pred_area += np.sum(masks_np)


#         # --- 2. 加载预测结果 ---
#     # print(f"Loading predictions from: {args.prediction_file}")
#     # file_ext = os.path.splitext(args.prediction_file)[1]
    
#     # total_pred_count = 0
#     # total_pred_area = 0.0
    
#     # if file_ext == '.pkl':
#     #     # --- 【关键修复】使用更健壮的 .pkl 解析逻辑 ---
#     #     try:
#     #         from mmdet.structures import DetDataSample
#     #     except ImportError:
#     #         print("Error: Failed to import DetDataSample...")
#     #         return

#     #     pred_data_samples = pickle.load(open(args.prediction_file, 'rb'))
#     #     print(f"Loaded {len(pred_data_samples)} data samples from .pkl file.")
        
#     #     # 遍历每个 data_sample
#     #     for data_sample in tqdm(pred_data_samples, desc="Processing PKL data"):
#     #         if not hasattr(data_sample, 'pred_instances'):
#     #             continue

#     #         pred_instances = data_sample.pred_instances
            
#     #         # 安全地获取 scores, labels, masks
#     #         scores = getattr(pred_instances, 'scores', None)
#     #         masks = getattr(pred_instances, 'masks', None)
            
#     #         if scores is None or masks is None or len(scores) == 0:
#     #             continue

#     #         # --- 应用置信度阈值过滤 ---
#     #         keep_indices = scores > args.score_thresh
            
#     #         # 获取过滤后的实例数量
#     #         num_kept = keep_indices.sum().item()
#     #         total_pred_count += num_kept

#     #         if num_kept > 0:
#     #             # 获取过滤后的掩码并计算面积
#     #             kept_masks = masks[keep_indices].to_ndarray()
#     #             total_pred_area += np.sum(kept_masks)

#     # --- 2. 加载预测结果 ---
#     print(f"Loading predictions from: {args.prediction_file}")
#     file_ext = os.path.splitext(args.prediction_file)[1]
    
#     preds_by_img = {} # key: img_id, value: list of pred dicts

#     if file_ext == '.pkl':
#         # --- 【关键修复】将 .pkl 文件当作一个包含了 COCO 格式预测结果的列表来处理 ---
#         with open(args.prediction_file, 'rb') as f:
#             # 加载进来的 pred_results 是一个列表
#             # 列表的每个元素，我们现在假设它是一个包含了 'img_id', 'pred_instances' 等键的字典
#             pred_data_samples = pickle.load(f)
#         print(f"Loaded {len(pred_data_samples)} data samples from .pkl file.")

#         # 获取 image_id 到尺寸的映射，以备解码 RLE
#         image_info_map = {img['id']: (img['height'], img['width']) for img in coco_gt.dataset['images']}

#         for sample in tqdm(pred_data_samples, desc="Parsing PKL data as dicts"):
#             # 假设 sample 是一个字典
#             if not isinstance(sample, dict): continue

#             # 从 sample 字典中获取 img_id
#             img_id = sample.get('img_id')
#             if img_id is None: continue

#             # 从 sample 字典中获取 pred_instances 字典
#             pred_instances = sample.get('pred_instances')
#             if not isinstance(pred_instances, dict): continue
            
#             # 从 pred_instances 字典中获取 scores 和 masks
#             scores = pred_instances.get('scores')
#             masks = pred_instances.get('masks') # 这里的 masks 是一个 RLE 列表

#             if scores is None or masks is None or len(scores) == 0: continue
            
#             if img_id not in preds_by_img:
#                 preds_by_img[img_id] = []

#             # 遍历每一个预测
#             for i in range(len(scores)):
#                 # 将 RLE 解码为二进制掩码
#                 h, w = image_info_map[img_id]
#                 rle = masks[i]
#                 if isinstance(rle, dict) and 'counts' in rle:
#                     try:
#                         mask = mask_utils.decode(rle)
#                         preds_by_img[img_id].append({
#                             'mask': mask,
#                             'score': scores[i]
#                         })
#                     except Exception as e:
#                         print(f"Warning: Failed to decode RLE for pred in image_id {img_id}. Error: {e}")

#     elif file_ext == '.json':
#         preds = json.load(open(args.prediction_file, 'r'))
#         print(f"Found {len(preds)} raw predictions in .json file.")
        
#         filtered_preds = [p for p in preds if p.get('score', 0) > args.score_thresh]
#         total_pred_count = len(filtered_preds)
#         print(f"Found {total_pred_count} predictions after applying score threshold > {args.score_thresh}")

#         image_info_map = {img['id']: (img['height'], img['width']) for img in coco_gt.dataset['images']}
        
#         for pred in tqdm(filtered_preds, desc="Calculating pred areas from JSON"):
#             if 'area' in pred:
#                 total_pred_area += pred['area']
#             elif 'segmentation' in pred:
#                 img_id = pred.get('image_id')
#                 if img_id is None or img_id not in image_info_map: continue
#                 rle = pred['segmentation']
#                 if isinstance(rle, dict) and 'counts' in rle:
#                     try:
#                         mask = mask_utils.decode(rle)
#                         total_pred_area += np.sum(mask)
#                     except Exception as e:
#                         print(f"Warning: Failed to decode RLE for pred in image_id {img_id}. Error: {e}")

#     else:
#         raise ValueError("Unsupported prediction file format.")

#     avg_pred_area = total_pred_area / total_pred_count if total_pred_count > 0 else 0

#     # --- 3. 计算差异并打印对比结果 ---
#     # 【关键修复】现在这里的变量 total_gt_count 和 avg_gt_area 都是被正确定义的
#     count_diff_percent = ((total_pred_count - total_gt_instances) / total_gt_instances) * 100
#     area_diff_percent = ((avg_pred_area - avg_gt_area) / avg_gt_area) * 100 if avg_gt_area > 0 else float('inf')

#     print("\n" + "="*70)
#     print(" " * 20 + "Instance Statistics Analysis")
#     print("="*70)
#     print(f"{'Metric':<30} | {'Ground Truth':<18} | {'Prediction':<18}")
#     print("-" * 70)
#     print(f"{'Total Instance Count':<30} | {total_gt_instances:<18} | {total_pred_count:<18}")
#     print(f"{'Average Instance Area (pixels)':<30} | {avg_gt_area:<18.2f} | {avg_pred_area:<18.2f}")
#     print("-" * 70)
#     print(f"{'Instance Count Difference (%)':<30} | {'-':<18} | {count_diff_percent:<+18.2f}%")
#     print(f"{'Average Area Difference (%)':<30} | {'-':<18} | {area_diff_percent:<+18.2f}%")
#     print("="*70)

#     print("\nInterpretation:")
#     if count_diff_percent < -5.0 and area_diff_percent > 5.0:
#         print("--> Conclusion: Prediction shows a significant tendency to MERGE instances.")
#     elif count_diff_percent > 5.0 and area_diff_percent < -5.0:
#         print("--> Conclusion: Prediction shows a significant tendency to SPLIT instances.")
#     else:
#         print("--> Conclusion: Prediction statistics are relatively consistent with ground truth.")

# if __name__ == '__main__':
#     main()



import argparse
import pickle
import json
import os
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

def calculate_iou(mask1, mask2):
    """
    计算两个二进制 NumPy 掩码的 IoU (Intersection over Union)。
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser(
        description='Calculate and compare instance statistics (count and area) between predictions and ground truth.')
    parser.add_argument(
        'prediction_file', 
        help='Path to the prediction file (.pkl from MMDetection or .json from YOLO).')
    parser.add_argument(
        'gt_file', 
        help='Path to the ground-truth COCO annotation file (.json). This file defines the scope of analysis.')
    parser.add_argument(
        '--score-thresh', 
        type=float, 
        default=0.3, 
        help='Confidence score threshold to filter predictions.')
    args = parser.parse_args()

    # --- 1. 加载 GT 文件来定义分析范围和计算基准 ---
    print(f"Loading ground truth from: {args.gt_file}")
    try:
        coco_gt = COCO(args.gt_file)
    except Exception as e:
        print(f"Error loading GT file: {e}")
        return
    
    # 从 GT 文件中获取我们关心的图片 ID 集合
    img_ids_to_analyze = set(coco_gt.getImgIds())
    if not img_ids_to_analyze:
        print("Error: The provided GT file contains no images.")
        return
    print(f"Analysis will be performed on {len(img_ids_to_analyze)} images defined by the GT file.")

    # 计算 GT 统计数据
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    total_gt_instances = len(gt_anns)
    if total_gt_instances == 0:
        print("Warning: The GT file contains images but no annotation instances.")
    
    total_gt_area = sum([ann.get('area', 0) for ann in gt_anns])
    avg_gt_area = total_gt_area / total_gt_instances if total_gt_instances > 0 else 0

    # --- 2. 加载预测结果，并解析成统一的内部格式 ---
    print(f"Loading predictions from: {args.prediction_file}")
    file_ext = os.path.splitext(args.prediction_file)[1]
    
    # 统一的中间数据结构: {img_id: [{'mask': ndarray, 'score': float}, ...]}
    preds_by_img = {}
    
    image_info_map = {img['id']: (img['height'], img['width']) for img in coco_gt.dataset['images']}

    if file_ext == '.pkl':
        # --- 将 .pkl 文件当作一个包含了 COCO 格式预测结果的列表来处理 ---
        with open(args.prediction_file, 'rb') as f:
            pred_data = pickle.load(f)
        print(f"Loaded {len(pred_data)} prediction items from .pkl file.")
        
        for pred_item in tqdm(pred_data, desc="Parsing PKL data as dicts"):
            if not isinstance(pred_item, dict): continue

            # 假设 pkl 里的每个 item 结构类似 COCO-JSON 的预测条目
            # 并且 img_id 是正确的整数 ID
            img_id = pred_item.get('image_id')
            if img_id is None or img_id not in img_ids_to_analyze:
                continue
            
            if img_id not in preds_by_img:
                preds_by_img[img_id] = []

            score = pred_item.get('score', 1.0)
            rle = pred_item.get('segmentation')
            
            if score > args.score_thresh and isinstance(rle, dict):
                try:
                    h, w = image_info_map[img_id]
                    mask = mask_utils.decode(rle)
                    preds_by_img[img_id].append({'mask': mask, 'score': score})
                except Exception as e:
                    print(f"Warning: Failed to decode RLE in PKL for image_id {img_id}. Error: {e}")

    elif file_ext == '.json':
        # --- 处理标准的 COCO 格式 .json 文件 ---
        with open(args.prediction_file, 'r') as f:
            pred_data = json.load(f)
        print(f"Loaded {len(pred_data)} predictions from .json file.")
        
        for pred in tqdm(pred_data, desc="Parsing JSON data"):
            img_id = pred.get('image_id')
            if img_id is None or img_id not in img_ids_to_analyze:
                continue
                
            if img_id not in preds_by_img:
                preds_by_img[img_id] = []

            score = pred.get('score', 1.0)
            rle = pred.get('segmentation')
            
            if score > args.score_thresh and isinstance(rle, dict):
                try:
                    h, w = image_info_map[img_id]
                    mask = mask_utils.decode(rle)
                    preds_by_img[img_id].append({'mask': mask, 'score': score})
                except Exception as e:
                    print(f"Warning: Failed to decode RLE in JSON for image_id {img_id}. Error: {e}")
    else:
        raise ValueError("Unsupported prediction file format.")

    # --- 3. 基于解析好的 preds_by_img 计算最终统计数据 ---
    total_pred_count = 0
    total_pred_area = 0.0
    
    print("\nCalculating final statistics...")
    for img_id, preds in preds_by_img.items():
        total_pred_count += len(preds)
        for pred in preds:
            total_pred_area += np.sum(pred['mask'])
                
    avg_pred_area = total_pred_area / total_pred_count if total_pred_count > 0 else 0

    # --- 4. 计算差异并打印对比结果 ---
    count_diff_percent = ((total_pred_count - total_gt_instances) / total_gt_instances) * 100 if total_gt_instances > 0 else 0
    area_diff_percent = ((avg_pred_area - avg_gt_area) / avg_gt_area) * 100 if avg_gt_area > 0 else float('inf')

    print("\n" + "="*70)
    print(" " * 15 + "Instance Statistics Analysis")
    print("="*70)
    print(f"{'Metric':<30} | {'Ground Truth (Subset)':<20} | {'Prediction (Subset)':<20}")
    print("-" * 70)
    print(f"{'Total Instance Count':<30} | {total_gt_instances:<20} | {total_pred_count:<20}")
    print(f"{'Average Instance Area (pixels)':<30} | {avg_gt_area:<20.2f} | {avg_pred_area:<20.2f}")
    print("-" * 70)
    print(f"{'Instance Count Difference (%)':<30} | {'-':<20} | {count_diff_percent:<+20.2f}%")
    print(f"{'Average Area Difference (%)':<30} | {'-':<20} | {area_diff_percent:<+20.2f}%")
    print("="*70)

    print("\nInterpretation:")
    if count_diff_percent < -5.0 and area_diff_percent > 5.0:
        print("--> Conclusion: Prediction shows a significant tendency to MERGE instances.")
    elif count_diff_percent > 5.0 and area_diff_percent < -5.0:
        print("--> Conclusion: Prediction shows a significant tendency to SPLIT instances.")
    else:
        print("--> Conclusion: Prediction statistics are relatively consistent with ground truth.")

if __name__ == '__main__':
    main()