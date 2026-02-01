import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import torch
import argparse
# from skimage.morphology import watershed

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import mmengine
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks 
from pycocotools import mask as mask_util # <-- 【新增】导入 pycocotools

# ==============================================================================
#           密集建筑物实例分割后处理脚本
# ==============================================================================
#
# 功能：
#   1. 加载 RTMDet-Ins 的实例分割预测结果 (.pkl)
#   2. 加载密度图模型的预测结果 (图片文件夹)
#   3. 对每个实例掩码，通过几何属性判断其是否为“可疑粘连区域”
#   4. 对可疑区域，结合密度图预测的中心点，使用分水岭算法进行分离
#   5. 保存最终优化后的分割结果
#
# ==============================================================================

# --- 1. 识别可疑区域的辅助函数 ---

def calculate_geometry_properties(mask):
    """计算单个掩码的面积、坚实度和范围度"""
    # 确保掩码是 uint8 类型
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0
        
    contour = contours[0]
    
    # 计算面积
    area = cv2.contourArea(contour)
    if area == 0:
        return 0, 0, 0
        
    # 计算凸包面积 (用于坚实度)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 计算外接矩形面积 (用于范围度)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    return area, solidity, extent

def is_suspicious(mask_np, area_thresh, solidity_thresh):
    """
    判断一个掩码是否为可疑粘连区域
    
    Args:
        mask_np (np.ndarray): 单个二进制掩码 (HxW)
        area_thresh (float): 面积阈值
        solidity_thresh (float): 坚实度阈值
        
    Returns:
        bool: 是否可疑
    """
    # 首先进行快速的面积判断
    area = np.sum(mask_np)
    if area < area_thresh: # 如果面积不够大，直接判定为不可疑
        return False, area
        
    # 对于大面积掩码，再进行更精细的形状判断
    _, solidity, _ = calculate_geometry_properties(mask_np)
    
    if solidity < solidity_thresh:
        return True, area # 面积大且形状不规则，判定为可疑
        
    return False, area


# --- 2. 后处理分离的核心函数 ---


def separate_masks(instance_preds, density_map_dir, area_thresh, solidity_thresh, peak_min_distance):
    """
    对一批预测结果进行后处理分离。
    修复了对 list 格式掩码的处理。
    """
    new_results = []
    
    for i, data_sample_dict in enumerate(tqdm(instance_preds, desc="Post-processing images")):
        if 'pred_instances' not in data_sample_dict or not data_sample_dict['pred_instances']:
            new_results.append(data_sample_dict) # 保留没有预测结果的样本
            continue
            
        pred_instances_dict = data_sample_dict['pred_instances']
        
        # --- 【关键修复 1】从其他地方获取 H 和 W ---
        # 尝试从 'ori_shape' 或 'img_shape' 获取，这些通常存在于 meta 信息中
        if 'ori_shape' in data_sample_dict:
            h, w = data_sample_dict['ori_shape']
        elif 'img_shape' in data_sample_dict:
            h, w = data_sample_dict['img_shape']
        else:
            # 如果都找不到，这是一个后备方案，但可能会不准确
            print(f"Warning: Cannot find shape info for sample {i}, skipping.")
            new_results.append(data_sample_dict)
            continue
        # -----------------------------------------------

        masks_list = pred_instances_dict['masks']
        scores = pred_instances_dict['scores']
        labels = pred_instances_dict['labels']
        
        # 如果掩码列表为空，直接跳到下一个
        if len(masks_list) == 0:
            new_results.append(data_sample_dict)
            continue

        # 加载对应的密度图预测结果
        img_path = data_sample_dict.get('img_path', f'unknown_image_{i}.png')
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        density_map_path = os.path.join(density_map_dir, f"{img_name}.png")
        
        if not os.path.exists(density_map_path):
            print(f"Warning: Density map not found for {img_name}, skipping.")
            new_results.append(data_sample_dict)
            continue
            
        density_map = cv2.imread(density_map_path, cv2.IMREAD_GRAYSCALE)
        if density_map.shape != (h, w):
            density_map = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_LINEAR)

        final_masks_rle = [] # <-- 我们最终保存为 RLE 格式
        final_scores = []
        final_labels = []

        for j in range(len(scores)):
            # --- 【关键修复 2】将单个 RLE 掩码解码为 NumPy 数组 ---
            # masks_list[j] 是一个 RLE 字典，例如 {'size': [h, w], 'counts': b'....'}
            single_rle_mask = masks_list[j]
            mask_np = mask_util.decode(single_rle_mask).astype(np.uint8)
            # ----------------------------------------------------
            
            suspicious, area = is_suspicious(mask_np, area_thresh, solidity_thresh)

            if suspicious and len(scores) > 1 : # 只有多于一个实例时，分离才有意义
                masked_density_map = density_map * mask_np
                coordinates = peak_local_max(masked_density_map, min_distance=peak_min_distance)
                
                if len(coordinates) > 1:
                    markers = np.zeros_like(mask_np, dtype=np.int32)
                    for k, (y, x) in enumerate(coordinates):
                        markers[y, x] = k + 1
                    
                    distance = ndimage.distance_transform_edt(mask_np)
                    separated_labels = watershed(-distance, markers, mask=mask_np)
                    
                    num_separated = len(np.unique(separated_labels)) - 1
                    if num_separated > 1:
                        for k in range(1, num_separated + 1):
                            new_mask_np = (separated_labels == k).astype(np.uint8)
                            # --- 【关键修复 3】将分离后的 NumPy 掩码重新编码为 RLE ---
                            new_mask_rle = mask_util.encode(np.asfortranarray(new_mask_np))
                            final_masks_rle.append(new_mask_rle)
                            # ----------------------------------------------------
                            final_scores.append(scores[j])
                            final_labels.append(labels[j])
                        continue
            
            # 如果不可疑，或者分水岭失败，直接保留原始的 RLE 掩码
            final_masks_rle.append(single_rle_mask)
            final_scores.append(scores[j])
            final_labels.append(labels[j])



        
            
        # # 将处理后的结果重新打包成与输入格式一致的字典
        # new_pred_instances_dict = {
        #     # 保存为 RLE 列表
        #     'masks': final_masks_rle,
        #     'scores': np.array(final_scores),
        #     'labels': np.array(final_labels)
        # }
        
        # # new_data_sample_dict = {
        # #     'pred_instances': new_pred_instances_dict,
        # #     'img_path': img_path,
        # #     'ori_shape': (h, w),
        # # }
        # # new_results.append(new_data_sample_dict)
        # # 2. 创建一个新的结果字典，先从原始样本中复制所有内容
        # new_data_sample_dict = data_sample_dict.copy()
        
        # # 3. 用我们新生成的 pred_instances 覆盖掉旧的
        # new_data_sample_dict['pred_instances'] = new_pred_instances_dict
        
        # new_results.append(new_data_sample_dict)


        # final_bboxes = []
        # for rle_mask in final_masks_rle:
        #     bbox_xywh = mask_util.toBbox(rle_mask)
        #     final_bboxes.append(bbox_xywh)
        final_bboxes = []
        for rle_mask in final_masks_rle:
            bbox_xywh = mask_util.toBbox(rle_mask)
            # --- 【关键修复】进行坐标格式转换 ---
            x, y, w, h = bbox_xywh
            bbox_xyxy = [x, y, x + w, y + h]
            final_bboxes.append(bbox_xyxy)

        # 将所有结果打包成一个字典
        new_pred_instances_dict = {
            'masks': final_masks_rle,
            'scores': torch.from_numpy(np.array(final_scores)), # <-- 转换为张量
            'labels': torch.from_numpy(np.array(final_labels)), # <-- 转换为张量
            'bboxes': torch.from_numpy(np.array(final_bboxes))  # <-- 转换为张量
        }
        
        # 创建一个新的结果字典，并复制元信息
        new_data_sample_dict = data_sample_dict.copy()
        new_data_sample_dict['pred_instances'] = new_pred_instances_dict
        
        new_results.append(new_data_sample_dict)
        # --- 修复结束 ---
        



        
    return new_results

def main():
    parser = argparse.ArgumentParser(description='Post-process instance segmentation results to separate adhesion.')
    
    # --- 【关键修复】将所有 add-argument 改为 add_argument ---
    parser.add_argument('instance_preds', help='Path to the instance segmentation prediction file (.pkl).')
    parser.add_argument('density_map_dir', help='Path to the directory of predicted density maps.')
    parser.add_argument('output_pkl', help='Path to save the post-processed prediction file (.pkl).')
    parser.add_argument('--area-thresh', type=float, default=5000.0, help='Area threshold to identify large masks.')
    parser.add_argument('--solidity-thresh', type=float, default=0.85, help='Solidity threshold for irregular shapes.')
    parser.add_argument('--peak-min-dist', type=int, default=10, help='Minimum distance between peaks in density map.')
    
    args = parser.parse_args()

    # # 加载原始预测结果
    # print(f"Loading original predictions from {args.instance_preds}...")
    # original_preds = mmengine.load(args.instance_preds)

    # # 执行后处理
    # print("Starting post-processing to separate adhered instances...")
    # processed_preds = separate_masks(
    #     original_preds, 
    #     args.density_map_dir, 
    #     args.area_thresh, 
    #     args.solidity_thresh,
    #     args.peak_min_dist
    # )

    # # 保存处理后的结果
    # mmengine.dump(processed_preds, args.output_pkl)
    # print(f"\nPost-processing finished! New predictions saved to {args.output_pkl}")
    # print("You can now use this new .pkl file with `tools/analysis_tools/instance_seg_eval.py` to evaluate the final performance.")
    print(f"Loading original predictions from {args.instance_preds}...")
    original_preds = mmengine.load(args.instance_preds)
    
    # --- 【关键修复 3】检查加载的数据类型 ---
    if not isinstance(original_preds, list) or not isinstance(original_preds[0], dict):
        raise TypeError("The prediction file is not a list of dictionaries as expected. "
                        "Please check the format of your .pkl file.")
    # ----------------------------------------

    print("Starting post-processing to separate adhered instances...")
    processed_preds = separate_masks(
    original_preds, 
    args.density_map_dir, 
    args.area_thresh, 
    args.solidity_thresh,
    args.peak_min_dist
    )
    
    mmengine.dump(processed_preds, args.output_pkl)
    print(f"\nPost-processing finished! New predictions saved to {args.output_pkl}")
    print("You can now use this new .pkl file with `tools/analysis_tools/instance_seg_eval.py` to evaluate the final performance.")

if __name__ == '__main__':
    main()