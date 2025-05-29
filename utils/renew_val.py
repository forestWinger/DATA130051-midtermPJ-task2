import os
import json

VAL_IMG_DIR = './split_data/val'
VAL_JSON_PATH = './split_data/annotations/val.json'
VAL_JSON_PATH_NEW = './split_data/annotations/val_cleaned.json'  # 新文件备份

# 1. 读取 val 文件夹下所有图片文件名
img_files = set(os.listdir(VAL_IMG_DIR))

# 2. 加载原始 val.json
with open(VAL_JSON_PATH, 'r') as f:
    data = json.load(f)

# 3. 过滤 images 列表，只保留实际存在图片的条目
images_new = []
valid_img_ids = set()
for img in data['images']:
    if img['file_name'] in img_files:
        images_new.append(img)
        valid_img_ids.add(img['id'])

print(f"实际图片数: {len(img_files)}")
print(f"标注中实际存在图片数: {len(images_new)}")

# 4. 过滤 annotations，仅保留 images_new 对应图片的注释
annotations_new = [anno for anno in data['annotations'] if anno['image_id'] in valid_img_ids]

# 5. 重建json并保存
data_new = data.copy()
data_new['images'] = images_new
data_new['annotations'] = annotations_new

with open(VAL_JSON_PATH_NEW, 'w') as f:
    json.dump(data_new, f)

print(f"已生成新的 val_cleaned.json，可以安全替换原 val.json 使用。")
