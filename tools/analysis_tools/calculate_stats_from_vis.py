# import cv2
# import numpy as np
# import os
# import argparse
# from tqdm import tqdm

# def analyze_image_stats(image_path, lower_color, upper_color, min_area=20):
#     """
#     分析单张可视化结果图，返回实例数量和总面积。

#     Args:
#         image_path (str): 预测结果图片的路径。
#         lower_color (np.array): 掩码颜色的下界 (BGR格式)。
#         upper_color (np.array): 掩码颜色的上界 (BGR格式)。
#         min_area (int): 认为是有效实例的最小轮廓面积。

#     Returns:
#         tuple: (该图的实例数量, 该图所有实例的总面积)
#     """
#     # 1. 读取图片
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Warning: Could not read image {image_path}")
#         return 0, 0.0

#     # 2. 颜色阈值分割，找到所有掩码区域
#     mask = cv2.inRange(image, lower_color, upper_color)

#     # 3. 查找轮廓
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     instance_count = 0
#     total_area = 0.0
    
#     # 4. 遍历所有找到的轮廓
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         # 过滤掉过小的、可能是噪声的轮廓
#         if area > min_area:
#             instance_count += 1
#             total_area += area
            
#     return instance_count, total_area


# def main():
#     parser = argparse.ArgumentParser(description='Calculate instance count and average area from visualization images.')
#     parser.add_argument('vis_dir', help='Path to the directory containing visualization images.')
#     parser.add_argument('--color', nargs=3, type=int, required=True, 
#                         help='The approximate BGR color of the masks. e.g., --color 128 0 0 for blue.')
#     parser.add_argument('--tolerance', type=int, default=60,
#                         help='Color tolerance for thresholding.')
#     parser.add_argument('--min-area', type=int, default=20,
#                         help='Minimum contour area to be considered as a valid instance.')
#     args = parser.parse_args()

#     # --- 1. 确定颜色范围 ---
#     base_color = np.array(args.color)
#     lower_bound = np.clip(base_color - args.tolerance, 0, 255)
#     upper_bound = np.clip(base_color + args.tolerance, 0, 255)
#     # 对于半透明颜色，可以稍微放宽非主色调通道的上限
#     for i in range(3):
#         if base_color[i] < 100: # 如果某个通道不是主色
#              upper_bound[i] = max(upper_bound[i], 120)

#     print("--- Instance Statistics Calculation Setup ---")
#     print(f"Analyzing images in: {args.vis_dir}")
#     print(f"Targeting BGR color: {base_color}")
#     print(f"Using color range (BGR): Lower={lower_bound}, Upper={upper_bound}")
#     print(f"Minimum instance area: {args.min_area} pixels")
#     print("-" * 45)

#     # --- 2. 遍历文件夹并累加统计数据 ---
#     grand_total_instances = 0
#     grand_total_area = 0.0
    
#     image_files = [f for f in os.listdir(args.vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))]
    
#     if not image_files:
#         print(f"Error: No image files found in the directory: {args.vis_dir}")
#         return

#     print(f"Found {len(image_files)} images to process...")
#     for filename in tqdm(image_files, desc="Analyzing images"):
#         image_path = os.path.join(args.vis_dir, filename)
#         count, area = analyze_image_stats(image_path, lower_bound, upper_bound, args.min_area)
#         grand_total_instances += count
#         grand_total_area += area

#     # --- 3. 计算并打印最终结果 ---
#     average_area = grand_total_area / grand_total_instances if grand_total_instances > 0 else 0.0
    
#     print("\n" + "="*45)
#     print("      Final Instance Statistics")
#     print("="*45)
#     print(f"{'Total Images Processed':<30}: {len(image_files)}")
#     print(f"{'Total Predicted Instances':<30}: {grand_total_instances}")
#     print(f"{'Average Instance Area (pixels)':<30}: {average_area:.2f}")
#     print("="*45)


# if __name__ == '__main__':
#     main()

# tools/analysis_tools/count_instances_from_vis.py (精确颜色范围版)

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def analyze_image_stats(image_path, lower_bgr, upper_bgr, min_area=20, debug=False):
    """
    分析单张可视化结果图，返回实例数量和总面积。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return 0, 0.0

    # 使用用户直接指定的 BGR 颜色范围进行阈值分割
    mask = cv2.inRange(image, lower_bgr, upper_bgr)

    # 调试功能，用于可视化掩码提取效果
    if debug:
        scale = 800 / max(image.shape[0], image.shape[1])
        debug_w, debug_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
        
        original_resized = cv2.resize(image, (debug_w, debug_h))
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_colored, (debug_w, debug_h))
        
        debug_image = np.hstack([original_resized, mask_resized])
        
        cv2.imshow(f"Debug: {os.path.basename(image_path)} | Press ESC to quit, any other key to continue", debug_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27: # ESC key
            return -1, -1.0

    # 查找轮廓并计算
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    total_area = sum([cv2.contourArea(cnt) for cnt in valid_contours])
    
    return len(valid_contours), total_area

def main():
    parser = argparse.ArgumentParser(description='Count instances and calculate average area from visualization images using precise color ranges.')
    parser.add_argument('vis_dir', help='Path to the directory containing visualization images.')
    # --- 【关键修改】使用 lower-color 和 upper-color 参数 ---
    parser.add_argument('--lower-color', nargs=3, type=int, required=True, 
                        help='The BGR lower bound of the mask color. e.g., --lower-color 45 50 50')
    parser.add_argument('--upper-color', nargs=3, type=int, required=True, 
                        help='The BGR upper bound of the mask color. e.g., --upper-color 75 255 255')
    # --------------------------------------------------------
    parser.add_argument('--min-area', type=int, default=20, 
                        help='Minimum contour area to be considered as a valid instance.')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode to visually inspect the thresholding mask.')
    args = parser.parse_args()

    # 将输入的列表转换为 NumPy 数组
    lower_bound = np.array(args.lower_color, dtype=np.uint8)
    upper_bound = np.array(args.upper_color, dtype=np.uint8)

    print("--- Instance Statistics Calculation Setup ---")
    print(f"Analyzing images in: {args.vis_dir}")
    print(f"Using precise BGR color range: Lower={lower_bound}, Upper={upper_bound}")
    print(f"Minimum instance area: {args.min_area} pixels")
    print("-" * 45)

    grand_total_instances = 0
    grand_total_area = 0.0
    
    image_files = [f for f in os.listdir(args.vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"Error: No image files found in the directory: {args.vis_dir}")
        return

    print(f"Found {len(image_files)} images to process...")
    for filename in tqdm(image_files, desc="Analyzing images"):
        image_path = os.path.join(args.vis_dir, filename)
        count, area = analyze_image_stats(image_path, lower_bound, upper_bound, args.min_area, args.debug)
        
        if count == -1: # 用户在调试模式下按了 ESC
            print("\nDebug mode exited by user.")
            return
            
        grand_total_instances += count
        grand_total_area += area

    average_area = grand_total_area / grand_total_instances if grand_total_instances > 0 else 0.0
    
    print("\n" + "="*45)
    print("      Final Instance Statistics")
    print("="*45)
    print(f"{'Total Images Processed':<30}: {len(image_files)}")
    print(f"{'Total Predicted Instances':<30}: {grand_total_instances}")
    print(f"{'Average Instance Area (pixels)':<30}: {average_area:.2f}")
    print("="*45)

if __name__ == '__main__':
    main()