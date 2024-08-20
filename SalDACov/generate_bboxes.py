import os
import cv2
import glob
import json
import torch
import numpy as np
import skimage.feature as ft
from joblib import Parallel, delayed

def generate_bboxes(lung_info):
    
    with open(lung_info) as f:
        infos = json.load(f)

    image = cv2.imread(infos["image_path"])
    image_name = infos["image_path"].split('/')[-1]
    print(infos["image_path"])

    left_lung = infos['left_lung']['bbox']
    right_lung = infos['right_lung']['bbox']
    
    cv2.rectangle(image, (left_lung[0], left_lung[1]),
                         (left_lung[2], left_lung[3]), (255, 0, 0), 2)
    
    cv2.rectangle(image, (right_lung[0], right_lung[1]),
                         (right_lung[2], right_lung[3]), (255, 0, 0), 2)
    
    cv2.imwrite(f'{new_json_path}/{image_name}', image)
    
#mode = 'dataset'
#mode = 'stylegan'
mode = 'stargan'

json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/lung_areas/'
new_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/bboxes/'

if os.path.isdir(new_json_path):
    os.system(f'rm -rf {new_json_path}')
os.mkdir(new_json_path)

lung_infos = glob.glob(f'{json_path}/*.json')

Parallel(n_jobs=-1)(delayed(generate_bboxes)(lung_info) for lung_info in lung_infos)
print('--- END ---')