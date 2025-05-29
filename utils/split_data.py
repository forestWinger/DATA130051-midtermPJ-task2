import os
import json
import numpy as np
import shutil
 
# 数据集路径
dataset_root = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(dataset_root, "VOC2012/JPEGImages")
annotations_path = os.path.join(dataset_root, "converted_data/voc12_trainval.json")
 
# 输出路径
output_root = os.path.join(dataset_root, "split_data")
os.makedirs(output_root, exist_ok=True)
 
# 读取annotations.json文件
with open(annotations_path, "r") as f:
    annotations_data = json.load(f)
 
# 提取images, annotations, categories
images = annotations_data["images"]
annotations = annotations_data["annotations"]
categories = annotations_data["categories"]

for image in images:
    image["file_name"] = image["file_name"].split("/")[-1]
 
# 随机打乱数据
np.random.shuffle(images)
 
# 训练集，验证集，测试集比例
train_ratio, val_ratio, test_ratio = 0.8, 0.2, 0.0
 
# 计算训练集，验证集，测试集的大小
num_images = len(images)
num_train = int(num_images * train_ratio)
num_test = int(num_images * test_ratio)
num_val = num_images - num_train - num_test
 
# 划分数据集
train_images = images[:num_train]
val_images = images[num_train:num_train + num_val]
if num_test > 0:
    test_images = images[num_train + num_val:]
 
# 分别为训练集、验证集和测试集创建子文件夹
train_folder = os.path.join(output_root, "train")
val_folder = os.path.join(output_root, "val")
test_folder = os.path.join(output_root, "test")
annotations_folder = os.path.join(output_root, "annotations")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)
 
# 将图片文件复制到相应的子文件夹
for img in train_images:
    shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(train_folder, img["file_name"]))
 
for img in val_images:
    shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(val_folder, img["file_name"]))
 
if num_test > 0:
    for img in test_images:
        shutil.copy(os.path.join(images_folder, img["file_name"]), os.path.join(test_folder, img["file_name"]))
 
# 根据图片id分配annotations
def filter_annotations(annotations, image_ids):
    return [ann for ann in annotations if ann["image_id"] in image_ids]
 
train_ann = filter_annotations(annotations, [img["id"] for img in train_images])
val_ann = filter_annotations(annotations, [img["id"] for img in val_images])
if num_test > 0:
    test_ann = filter_annotations(annotations, [img["id"] for img in test_images])
 
# 生成train.json, val.json, test.json
train_json = {"images": train_images, "annotations": train_ann, "categories": categories}
val_json = {"images": val_images, "annotations": val_ann, "categories": categories}
if num_test > 0:
    test_json = {"images": test_images, "annotations": test_ann, "categories": categories}
 
with open(os.path.join(output_root, "annotations/train.json"), "w") as f:
    json.dump(train_json, f)
 
with open(os.path.join(output_root, "annotations/val.json"), "w") as f:
    json.dump(val_json, f)

if num_test > 0:
    with open(os.path.join(output_root, "annotations/test.json"), "w") as f:
        json.dump(test_json, f)
 
print("数据集划分完成！")