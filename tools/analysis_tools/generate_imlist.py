import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate an image list file from a COCO JSON.')
    parser.add_argument('gt_json', help='Path to the COCO JSON file.')
    parser.add_argument('output_txt', help='Path to save the output image list .txt file.')
    args = parser.parse_args()

    with open(args.gt_json, 'r') as f:
        data = json.load(f)
    
    with open(args.output_txt, 'w') as f:
        for image_info in data['images']:
            # 只写入文件名，不包含路径
            f.write(image_info['file_name'] + '\n')
            
    print(f"Successfully created image list at: {args.output_txt}")

if __name__ == '__main__':
    main()