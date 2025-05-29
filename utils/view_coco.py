import cv2
import os
import numpy as np
from pycocotools.coco import COCO


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

img_path = 'split_data/val'
annFile = 'split_data/annotations/val.json'
save_path = 'vis_val/images_vis'

if not os.path.exists(save_path):
    os.makedirs(save_path)


def draw_rectangle(coordinates, image, image_name):
    for coordinate in coordinates:
        left, top, right, bottom, label = map(int, coordinate[0:5])
        cat_name = coordinate[5]
        color = colors[label % len(colors)]
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, cat_name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.imwrite(save_path + '/' + image_name, image)


coco = COCO(annFile)

catIds = coco.getCatIds(catNms=['Crack','Manhole', 'Net', 'Pothole','Patch-Crack', "Patch-Net", "Patch-Pothole", "other"])
catIds = coco.getCatIds()
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds()

for imgId in imgIds:

    img = coco.loadImgs(imgId)[0]
    image_name = img['file_name']
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    for ann in anns:
    # 将类别编号转换为类别名称
        category_id = ann['category_id']
        category_info = coco.loadCats(category_id)[0]
        category_name = category_info['name']
        ann['category_name'] = category_name
    coco.showAnns(anns)
    
    coordinates = []
    img_raw = cv2.imread(os.path.join(img_path, image_name))
    for j in range(len(anns)):
        coordinate = anns[j]['bbox']
        coordinate[2] += coordinate[0]
        coordinate[3] += coordinate[1]
        coordinate.append(anns[j]['category_id'])
        coordinate.append(anns[j]['category_name'])
        coordinates.append(coordinate)

    draw_rectangle(coordinates, img_raw, image_name)
