# DATA130051-midtermPJ-task2
25springCV期中pj任务2 目标检测

use utils/split_data.py to split the original dataset
训练：python ./mmdetection/tools/train.py config1.py
测试：python ./mmdetection/demo/image_demo.py 4.jpg work_dirs/config1/epoch_24.pth --out-dir vision_results/toy
