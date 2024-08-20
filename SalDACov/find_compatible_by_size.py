import os
import sys
import cv2
import glob
import json
import random
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def find_compatibles(gan_lung_info):

    dataset_lung_infos = glob.glob(f'{dataset_json_path}/*.json')
    
    with open(gan_lung_info) as f:
        gan_infos = json.load(f)

    #print(gan_lung_info)
    #print(gan_infos)

    # Get left lung width, min and max range
    gan_left_lung_width = gan_infos['left_lung']['bbox'][2] - gan_infos['left_lung']['bbox'][0]
    gan_min_left_lung_width = int(0.9 * gan_left_lung_width)
    gan_max_left_lung_width = int(gan_left_lung_width + 0.1 * gan_left_lung_width)

    # Get left lung height, min and max range
    gan_left_lung_height = gan_infos['left_lung']['bbox'][3] - gan_infos['left_lung']['bbox'][1]
    gan_min_left_lung_height = int(0.9 * gan_left_lung_height)
    gan_max_left_lung_height = int(gan_left_lung_height + 0.1 * gan_left_lung_height)

    # Get right lung width, min and max range
    gan_right_lung_width = gan_infos['right_lung']['bbox'][2] - gan_infos['right_lung']['bbox'][0]
    gan_min_right_lung_width = int(0.9 * gan_right_lung_width)
    gan_max_right_lung_width = int(gan_right_lung_width + 0.1 * gan_right_lung_width)

    # Get right lung height, min and max range
    gan_right_lung_height = gan_infos['right_lung']['bbox'][3] - gan_infos['right_lung']['bbox'][1] 
    gan_min_right_lung_height = int(0.9 * gan_right_lung_height)
    gan_max_right_lung_height = int(gan_right_lung_height + 0.1 * gan_right_lung_height)

    compatibles_left_left = []
    compatibles_right_right = []
    compatibles_left_right = []
    compatibles_right_left = []
    
    random.shuffle(dataset_lung_infos)
    for dataset_lung_info in dataset_lung_infos:

        with open(dataset_lung_info) as f:
            dataset_infos = json.load(f)
        
        #print(dataset_infos["lesion_mask_path"])
        lesion_mask = cv2.imread(dataset_infos["lesion_mask_path"], 0)
        lesion_mask = np.where(lesion_mask != 0, 1, 0)        
        
        left_mask = lesion_mask[dataset_infos['left_lung']['bbox'][1]:dataset_infos['left_lung']['bbox'][3],
                                dataset_infos['left_lung']['bbox'][0]:dataset_infos['left_lung']['bbox'][2]]
        right_mask = lesion_mask[dataset_infos['right_lung']['bbox'][1]:dataset_infos['right_lung']['bbox'][3],
                                 dataset_infos['right_lung']['bbox'][0]:dataset_infos['right_lung']['bbox'][2]]

        left_area = np.sum(left_mask == 1)
        right_area = np.sum(right_mask == 1)
        
        '''
        print(left_area)
        print(right_area)

        lesion_mask = np.where(lesion_mask != 0, 255, 0)
        cv2.imwrite('lesion_mask.png', lesion_mask)
        left_mask = np.where(left_mask != 0, 255, 0)
        right_mask = np.where(right_mask != 0, 255, 0)
        cv2.imwrite('lesion_mask_left.png', left_mask)
        cv2.imwrite('lesion_mask_right.png', right_mask)
        break
        '''

        dataset_left_width = dataset_infos['left_lung']['bbox'][2] - dataset_infos['left_lung']['bbox'][0]
        dataset_left_height = dataset_infos['left_lung']['bbox'][3] - dataset_infos['left_lung']['bbox'][1]

        dataset_right_width = dataset_infos['right_lung']['bbox'][2] - dataset_infos['right_lung']['bbox'][0] 
        dataset_right_height = dataset_infos['right_lung']['bbox'][3] - dataset_infos['right_lung']['bbox'][1]
        
        if left_area > 1000:
            # left lesion - gan left lung
            if dataset_left_width >= gan_min_left_lung_width and dataset_left_width <= gan_max_left_lung_width and \
                dataset_left_height >= gan_min_left_lung_height and dataset_left_height <= gan_max_left_lung_height:
                compatibles_left_left.append(dataset_lung_info)

            # left lesion - gan right lung
            if dataset_left_width >= gan_min_right_lung_width and dataset_left_width <= gan_max_right_lung_width and \
                dataset_left_height >= gan_min_right_lung_height and dataset_left_height <= gan_max_right_lung_height:
                compatibles_right_left.append(dataset_lung_info)
        
        if right_area > 1000:
            #right lesion - gan right lung
            if dataset_right_width >= gan_min_right_lung_width and dataset_right_width <= gan_max_right_lung_width and \
                dataset_right_height >= gan_min_right_lung_height and dataset_right_height <= gan_max_right_lung_height:
                compatibles_right_right.append(dataset_lung_info)

            # right lesion - gan left lung
            if dataset_right_width >= gan_min_left_lung_width and dataset_right_width <= gan_max_left_lung_width and \
                dataset_right_height >= gan_min_left_lung_height and dataset_right_height <= gan_max_left_lung_height:
                compatibles_left_right.append(dataset_lung_info)

        if len(compatibles_left_left) >= 100 and len(compatibles_left_right) >= 100 and \
            len(compatibles_right_right) >= 100 and len(compatibles_right_left) >= 100:
            break

    gan_infos['left_lung']['compatibles_left'] = compatibles_left_left
    gan_infos['left_lung']['compatibles_right'] = compatibles_left_right

    gan_infos['right_lung']['compatibles_right'] = compatibles_right_right
    gan_infos['right_lung']['compatibles_left'] = compatibles_right_left

    #print(len(compatibles_left))
    #print(len(compatibles_left_flipped))
    #print(len(compatibles_right))
    #print(len(compatibles_right_flipped))

    if len(compatibles_left_left) >= 25 and len(compatibles_left_right) >= 25 and \
        len(compatibles_right_right) >= 25 and len(compatibles_right_left) >= 25:

        json_name = gan_lung_info.split('/')[-1]
        #print(f'{compatibles_json_path}/{json_name}')
        with open(f'{compatibles_json_path}{json_name}', "w") as outfile:
            json.dump(gan_infos, outfile, indent=2)

gan = sys.argv[1]

gan_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/features/'
dataset_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/dataset/features/'
compatibles_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/size_compatibles/'

if os.path.isdir(compatibles_json_path):
    os.system(f'rm -rf {compatibles_json_path}')
os.mkdir(compatibles_json_path)

gan_lung_infos = glob.glob(f'{gan_json_path}/*.json')

#find_compatibles(gan_lung_infos[0])

Parallel(n_jobs=-1, verbose=10)(delayed(find_compatibles)(gan_lung_info) for gan_lung_info in gan_lung_infos)
print('--- END ---')