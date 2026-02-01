# import os
# import argparse
# import re
# from tqdm import tqdm

# def rename_density_maps(density_dir, original_image_names):
#     """
#     批量重命名密度图文件，使其与原始图像名匹配。

#     Args:
#         density_dir (str): 存放密度图的文件夹路径。
#         original_image_names (list): 包含所有原始图像文件名（不带扩展名）的列表。
#     """
#     print(f"Scanning directory: {density_dir}")
#     files_to_rename = os.listdir(density_dir)
#     rename_count = 0
    
#     # 创建一个快速查找集合，提高效率
#     original_names_set = set(original_image_names)

#     for old_filename in tqdm(files_to_rename, desc="Renaming files"):
#         # 我们要从 'test_pk_13.tif_48.png' 中，智能地提取出原始图像名
        
#         found_original_name = None
        
#         # 策略1：直接在文件名中查找匹配的原始图像名
#         # 这种方法最鲁棒，可以处理 'wh3-82-47' 这种情况
#         for original_name in original_names_set:
#             if original_name in old_filename:
#                 found_original_name = original_name
#                 break
        
#         if found_original_name:
#             # 构建新的文件名
#             extension = os.path.splitext(old_filename)[1] # 获取扩展名，如 .png
#             new_filename = f"{found_original_name}{extension}"
            
#             old_filepath = os.path.join(density_dir, old_filename)
#             new_filepath = os.path.join(density_dir, new_filename)

#             # 如果新文件名与旧文件名不同，且新文件名不存在，则重命名
#             if old_filepath != new_filepath:
#                 if not os.path.exists(new_filepath):
#                     os.rename(old_filepath, new_filepath)
#                     rename_count += 1
#                 else:
#                     print(f"Warning: Target file '{new_filepath}' already exists. Skipping rename for '{old_filename}'.")
#         else:
#             print(f"Warning: Could not find a matching original image name for '{old_filename}'. Skipping.")
            
#     print(f"\nFinished! Renamed {rename_count} files.")


# def get_original_names_from_pkl(prediction_pkl):
#     """从预测的 .pkl 文件中提取所有原始图像的文件名。"""
#     import pickle
#     print(f"Extracting original image names from {prediction_pkl}...")
#     preds_data = pickle.load(open(prediction_pkl, 'rb'))
    
#     original_names = []
#     for data_sample in preds_data:
#         # 兼容 dict 和 DataSample 两种格式
#         if isinstance(data_sample, dict):
#             img_path = data_sample.get('img_path', '')
#         else: # 假设是 DetDataSample 或类似对象
#             img_path = getattr(data_sample, 'img_path', '')
            
#         if img_path:
#             # 提取不带扩展名的文件名
#             base_name = os.path.splitext(os.path.basename(img_path))[0]
#             original_names.append(base_name)
            
#     if not original_names:
#         raise ValueError("Could not extract any image names from the .pkl file. "
#                          "Please ensure 'img_path' key/attribute exists.")
                         
#     print(f"Found {len(original_names)} unique image names.")
#     return list(set(original_names))


# def main():
#     parser = argparse.ArgumentParser(description='Batch rename density map prediction files.')
#     parser.add_argument('density_dir', help='Path to the directory of predicted density maps.')
#     parser.add_argument('prediction_pkl', help='Path to the instance segmentation prediction .pkl file, used to get original image names.')
    
#     args = parser.parse_args()
    
#     try:
#         original_names = get_original_names_from_pkl(args.prediction_pkl)
#         rename_density_maps(args.density_dir, original_names)
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")

# if __name__ == '__main__':
#     main()


import os
import argparse
from tqdm import tqdm

def rename_density_maps(density_dir, original_image_names):
    print(f"Scanning directory: {density_dir}")
    files_to_rename = os.listdir(density_dir)
    rename_count = 0
    
    original_names_set = set(original_image_names)

    for old_filename in tqdm(files_to_rename, desc="Renaming files"):
        
        # --- 【关键修复】使用“最长匹配”策略 ---
        best_match = None
        # 遍历所有可能的原始文件名
        for original_name in original_names_set:
            if original_name in old_filename:
                # 如果找到了一个匹配
                if best_match is None or len(original_name) > len(best_match):
                    # 如果这是第一个匹配，或者这个匹配比之前的更长，就更新它
                    best_match = original_name
        
        if best_match:
            extension = os.path.splitext(old_filename)[1]
            new_filename = f"{best_match}{extension}"
            
            old_filepath = os.path.join(density_dir, old_filename)
            new_filepath = os.path.join(density_dir, new_filename)

            if old_filepath != new_filepath:
                if not os.path.exists(new_filepath):
                    os.rename(old_filepath, new_filepath)
                    rename_count += 1
                else:
                    # 如果目标文件已存在，但不是由当前文件重命名的，说明有冲突
                    # 我们可以选择覆盖或打印更详细的警告
                    print(f"Warning: Target file '{new_filename}' already exists. "
                          f"Could not rename '{old_filename}'. Possible naming conflict.")
        else:
            print(f"Warning: Could not find a matching original image name for '{old_filename}'.")
            
    print(f"\nFinished! Renamed {rename_count} files.")

def get_original_names_from_pkl(prediction_pkl):
    """从预测的 .pkl 文件中提取所有原始图像的文件名。"""
    import pickle
    print(f"Extracting original image names from {prediction_pkl}...")
    preds_data = pickle.load(open(prediction_pkl, 'rb'))
    
    original_names = []
    for data_sample in preds_data:
        # 兼容 dict 和 DataSample 两种格式
        if isinstance(data_sample, dict):
            img_path = data_sample.get('img_path', '')
        else: # 假设是 DetDataSample 或类似对象
            img_path = getattr(data_sample, 'img_path', '')
            
        if img_path:
            # 提取不带扩展名的文件名
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            original_names.append(base_name)
            
    if not original_names:
        raise ValueError("Could not extract any image names from the .pkl file. "
                         "Please ensure 'img_path' key/attribute exists.")
                         
    print(f"Found {len(original_names)} unique image names.")
    return list(set(original_names))


def main():
    parser = argparse.ArgumentParser(description='Batch rename density map prediction files.')
    parser.add_argument('density_dir', help='Path to the directory of predicted density maps.')
    parser.add_argument('prediction_pkl', help='Path to the instance segmentation prediction .pkl file, used to get original image names.')
    
    args = parser.parse_args()
    
    try:
        original_names = get_original_names_from_pkl(args.prediction_pkl)
        rename_density_maps(args.density_dir, original_names)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == '__main__':
    main()