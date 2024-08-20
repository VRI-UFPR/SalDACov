import os
import sys
import cv2
import glob
import json
import torch
import numpy as np
import skimage.feature as ft
from joblib import Parallel, delayed

NBINS = 128
DEVICE = "cpu"
def generate_hist(image):
    hist = torch.histc(torch.from_numpy(image).float().to(DEVICE), bins=NBINS, min=0.0, max=255.0)

    #print(hist)
    
    lbp = ft.local_binary_pattern(image, 24, 3, 'uniform')
    lbp_hist = torch.histc(torch.from_numpy(lbp).float().to(DEVICE), bins=NBINS, min=0.0, max=255.0)

    #print(lbp_hist)
    
    hist = torch.stack((hist, lbp_hist))
    hist = hist.view(-1)
    return hist.tolist()

def generate_features(lung_info):
    
    with open(lung_info) as f:
        infos = json.load(f)

    image = cv2.imread(infos["image_path"], 0)
    image_name = infos["image_path"].split('/')[-1]
    #print(infos["image_path"])

    if mode == 'dataset':
        mask = cv2.imread(infos["lesion_mask_path"], 0)    
    else:
        mask = cv2.imread(infos["lung_mask_path"], 0)

    mask = np.where(mask != 0, 1, 0)
    masked = image * mask

    #print(infos['left_lung'])
    #print(infos['right_lung'])

    left_lung = masked[infos['left_lung']['bbox'][1]:infos['left_lung']['bbox'][3],
                       infos['left_lung']['bbox'][0]:infos['left_lung']['bbox'][2]]
    
    right_lung = masked[infos['right_lung']['bbox'][1]:infos['right_lung']['bbox'][3],
                        infos['right_lung']['bbox'][0]:infos['right_lung']['bbox'][2]]
    
    #cv2.imwrite('image.jpg', masked)
    #cv2.imwrite('left.jpg', left_lung)
    #cv2.imwrite('right.jpg', right_lung)

    left_hist = generate_hist(left_lung)
    right_hist = generate_hist(right_lung)

    #print(left_hist)
    #print(right_hist)
    
    infos['left_lung']['features'] = left_hist
    infos['right_lung']['features'] = right_hist

    #print(infos)

    json_name = image_name.replace('.jpg','.json')
    #print(json_name)
    with open(f'{new_json_path}/{json_name}', "w") as outfile:
        json.dump(infos, outfile, indent=2)
    
#mode = 'dataset'
#mode = 'stylegan'
#mode = 'stargan'
mode = sys.argv[1]

json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/lung_areas/'
new_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/features/'

if os.path.isdir(new_json_path):
    os.system(f'rm -rf {new_json_path}')
os.mkdir(new_json_path)

lung_infos = glob.glob(f'{json_path}/*.json')

#generate_features(lung_infos[0])

Parallel(n_jobs=-1, verbose=10)(delayed(generate_features)(lung_info) for lung_info in lung_infos)
print('--- END ---')