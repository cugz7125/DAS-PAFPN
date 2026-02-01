# import cv2
# import numpy as np
# import os
# import argparse
# from tqdm import tqdm

# def main():
#     parser = argparse.ArgumentParser(description='Count instances from visualized prediction images based on color.')
#     parser.add_argument('vis_dir', help='Directory containing the visualized prediction images.')
    
#     # --- 【关键】你需要在这里定义颜色的 HSV 范围 ---
#     # 这个范围需要你手动确定。我提供了一个常见的红色范围作为例子。
#     # H (色调): 0-180, S (饱和度): 0-255, V (明度): 0-255
#     parser.add_argument('--lower-color', nargs=3, type=int, default=[0, 100, 100], 
#                         help='Lower bound of the color to segment in HSV. Format: H S V')
#     parser.add_argument('--upper-color', nargs=3, type=int, default=[10, 255, 255],
#                         help='Upper bound of the color to segment in HSV. Format: H S V')
    
#     parser.add_argument('--min-area', type=int, default=50, 
#                         help='Minimum pixel area to be considered as a valid instance.')

#     args = parser.parse_args()

#     # 将列表转换为 NumPy 数组
#     lower_color = np.array(args.lower_color)
#     upper_color = np.array(args.upper_color)

#     if not os.path.isdir(args.vis_dir):
#         print(f"Error: Directory not found at '{args.vis_dir}'")
#         return

#     image_files = [f for f in os.listdir(args.vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
#     if not image_files:
#         print(f"Error: No image files found in '{args.vis_dir}'")
#         return

#     total_instance_count = 0
#     print(f"Found {len(image_files)} images to process in '{args.vis_dir}'...")

#     for image_name in tqdm(image_files):
#         image_path = os.path.join(args.vis_dir, image_name)
        
#         # 1. 读取图片
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Warning: Could not read image {image_name}. Skipping.")
#             continue
            
#         # 2. 将图片从 BGR 转换到 HSV 色彩空间
#         # HSV 空间对光照变化不那么敏感，更容易根据颜色进行分割
#         hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         # 3. 根据颜色范围创建二值掩码
#         # 所有在 [lower_color, upper_color] 范围内的像素都会变成白色 (255)
#         color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

#         # 4. 使用连通域分析来计数
#         # num_labels: 连通域的总数（包括背景）
#         # labels: 一个与原图等大的矩阵，每个连通域被赋予一个唯一的整数标签
#         # stats: 每个连通域的统计信息 [x, y, width, height, area]
#         # centroids: 每个连通域的中心点
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

#         # 连通域的数量包含了背景（标签0），所以实际实例数是 num_labels - 1
#         # 我们还需要根据最小面积进行过滤，去除小的噪声点
#         instance_count_in_image = 0
#         for i in range(1, num_labels): # 从1开始，跳过背景
#             if stats[i, cv2.CC_STAT_AREA] >= args.min_area:
#                 instance_count_in_image += 1
        
#         total_instance_count += instance_count_in_image

#     print("\n" + "="*40)
#     print("      Instance Counting Results")
#     print("="*40)
#     print(f"Directory Analyzed: {args.vis_dir}")
#     print(f"Color Range (HSV): Lower={args.lower_color}, Upper={args.upper_color}")
#     print(f"Minimum Instance Area: {args.min_area} pixels")
#     print("-" * 40)
#     print(f"Total Predicted Instances Found: {total_instance_count}")
#     print("="*40)

# if __name__ == '__main__':
#     main()




# count_instances_from_images_with_stats.py (最终版)

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description='Count instances and calculate average area from visualized prediction images using HSV color space.')
    
    parser.add_argument('vis_dir', help='Directory containing the visualized prediction images.')
    
    # --- 参数保持与之前效果最好的脚本一致 ---
    # H (色调): 0-180, S (饱和度): 0-255, V (明度): 0-255
    parser.add_argument('--lower-color', nargs=3, type=int, required=True, 
                        help='Lower bound of the color to segment in HSV. Format: H S V')
    parser.add_argument('--upper-color', nargs=3, type=int, required=True,
                        help='Upper bound of the color to segment in HSV. Format: H S V')
    
    parser.add_argument('--min-area', type=int, default=50, 
                        help='Minimum pixel area to be considered as a valid instance.')
    
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode to visually inspect the color mask.')

    args = parser.parse_args()

    # 将列表转换为 NumPy 数组
    lower_color = np.array(args.lower_color)
    upper_color = np.array(args.upper_color)

    if not os.path.isdir(args.vis_dir):
        print(f"Error: Directory not found at '{args.vis_dir}'")
        return

    image_files = [f for f in os.listdir(args.vis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"Error: No image files found in '{args.vis_dir}'")
        return

    # --- 初始化总数和总面积的累加器 ---
    grand_total_instances = 0
    grand_total_area = 0.0

    print(f"Found {len(image_files)} images to process in '{args.vis_dir}'...")

    for image_name in tqdm(image_files, desc="Analyzing images"):
        image_path = os.path.join(args.vis_dir, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_name}. Skipping.")
            continue
            
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

        if args.debug:
            # 可视化调试
            scale = 800 / max(image.shape[0], image.shape[1])
            debug_w, debug_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
            original_resized = cv2.resize(image, (debug_w, debug_h))
            mask_colored = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
            mask_resized = cv2.resize(mask_colored, (debug_w, debug_h))
            debug_image = np.hstack([original_resized, mask_resized])
            cv2.imshow(f"Debug: {image_name} | Press ESC to quit", debug_image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                print("\nDebug mode exited by user.")
                return

        # 使用连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)

        # --- 【关键修改】在计数的循环中，同时累加面积 ---
        for i in range(1, num_labels): # 从1开始，跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= args.min_area:
                grand_total_instances += 1
                grand_total_area += area # <-- 新增：累加面积
        # -----------------------------------------------
    
    # 计算平均面积
    average_area = grand_total_area / grand_total_instances if grand_total_instances > 0 else 0.0

    print("\n" + "="*45)
    print("      Instance Statistics Results")
    print("="*45)
    print(f"Directory Analyzed: {args.vis_dir}")
    print(f"Color Range (HSV): Lower={args.lower_color}, Upper={args.upper_color}")
    print(f"Minimum Instance Area: {args.min_area} pixels")
    print("-" * 45)
    print(f"{'Total Predicted Instances':<30}: {grand_total_instances}")
    print(f"{'Average Instance Area (pixels)':<30}: {average_area:.2f}") # <-- 新增：打印平均面积
    print("="*45)

if __name__ == '__main__':
    main()