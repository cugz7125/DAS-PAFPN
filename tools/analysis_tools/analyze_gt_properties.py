# tools/analysis_tools/analyze_gt_properties.py (修正版)

import json
import numpy as np
import cv2
from pycocotools import mask as mask_util
from tqdm import tqdm
import argparse

def get_gt_properties(ann_file):
    """
    分析 COCO 格式标注文件中所有实例的几何属性。
    修复了从标注中直接获取图像尺寸的 KeyError。
    """
    print(f"Loading annotation file: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
        
    annotations = coco_data.get('annotations', [])
    images_info = coco_data.get('images', [])
    
    if not annotations or not images_info:
        print("Annotations or images info not found in the file.")
        return np.array([]), np.array([])

    # --- 【关键修复】创建一个从 image_id 到尺寸的映射字典 ---
    image_id_to_shape = {img['id']: (img['height'], img['width']) for img in images_info}
    # --------------------------------------------------------

    areas = []
    solidities = []
    
    print("Analyzing ground truth masks...")
    for ann in tqdm(annotations):
        image_id = ann.get('image_id')
        if image_id is None or image_id not in image_id_to_shape:
            continue # 如果标注没有对应的图片信息，则跳过
            
        # --- 【关键修复】从映射字典中获取正确的图像高和宽 ---
        h, w = image_id_to_shape[image_id]
        # ----------------------------------------------------
        
        # COCO 格式的分割标注可能是多边形或 RLE
        if isinstance(ann['segmentation'], list):
            # 将多边形转换为 RLE (Run-Length Encoding)
            rle = mask_util.frPyObjects(ann['segmentation'], h, w)
            # 从 RLE 解码为二进制掩码
            binary_mask = mask_util.decode(rle)
            # 处理一个物体由多个不相连部分组成的情况
            if len(binary_mask.shape) > 2:
                binary_mask = np.sum(binary_mask, axis=2)
            
            binary_mask = (binary_mask > 0).astype(np.uint8)
        else: # 如果已经是 RLE 格式
            rle = ann['segmentation']
            binary_mask = mask_util.decode(rle).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # 只处理最大的那个轮廓（通常只有一个）
        contour = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(contour)
        if area <= 1: # 忽略面积过小的标注
            continue
            
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        areas.append(area)
        solidities.append(solidity)
        
    return np.array(areas), np.array(solidities)

def main():
    parser = argparse.ArgumentParser(description='Analyze geometry properties of ground truth masks.')
    parser.add_argument('ann_file', help='Path to the COCO format annotation file (e.g., instances_train2017.json).')
    args = parser.parse_args()

    areas, solidities = get_gt_properties(args.ann_file)
    
    if areas.size == 0:
        print("Could not analyze any valid masks.")
        return

    print("\n--- Geometry Properties Analysis Results ---")
    
    # 面积分析
    print("\n[Area Analysis]")
    print(f"  - Total valid instances: {len(areas)}")
    print(f"  - Mean Area: {np.mean(areas):.2f} pixels")
    print(f"  - Median Area: {np.median(areas):.2f} pixels")
    print(f"  - Std Dev of Area: {np.std(areas):.2f}")
    p90_area = np.percentile(areas, 90)
    p95_area = np.percentile(areas, 95)
    print(f"  - 90th Percentile Area: {p90_area:.2f} (90% of buildings are smaller than this)")
    print(f"  - 95th Percentile Area: {p95_area:.2f} (95% of buildings are smaller than this)")
    print(f"  => Recommended --area-thresh: {p95_area:.0f}")

    # 坚实度分析
    print("\n[Solidity Analysis]")
    print(f"  - Mean Solidity: {np.mean(solidities):.3f}")
    print(f"  - Median Solidity: {np.median(solidities):.3f}")
    p10_solidity = np.percentile(solidities, 10)
    p5_solidity = np.percentile(solidities, 5)
    print(f"  - 10th Percentile Solidity: {p10_solidity:.3f} (10% of buildings are more irregular than this)")
    print(f"  - 5th Percentile Solidity: {p5_solidity:.3f} (5% of buildings are more irregular than this)")
    print(f"  => Recommended --solidity-thresh: {p10_solidity:.2f}")
    
    print("\n-----------------------------------------")

if __name__ == '__main__':
    main()