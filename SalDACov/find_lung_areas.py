import os
import sys
import cv2
import glob
import json
import numpy as np
from joblib import Parallel, delayed

def find_lung_areas(image_path, mask_path):

    #print(image_path)
    #print(mask_path)
    
    infos = {
        "image_path": image_path,
        "lung_mask_path": mask_path,
    }

    if mode == 'dataset':
        infos["lesion_mask_path"] = mask_path.replace('/lungs/','/masks/')

    json_name = image_path.split('/')[-1].replace('.jpg','.json')

    #image = cv2.imread(image_path)
    mask = cv2.imread(mask_path,0)

    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hiers = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = []
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        lung = mask[y:y+h, x:x+w]
        lung = np.where(lung != 0, 1, 0)
        area = int(np.sum(lung == 1))
        if area > 10000:
            areas.append(area)
            bboxes.append((x, y, x + w, y + h))
            #image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    #print(bboxes)
    #print(bboxes[0][0])
    #print(bboxes[1][0])

    left_lung = {}
    right_lung = {}    
    if len(areas) == 2:
        if bboxes[0][0] > bboxes[1][0]:
            left_lung['bbox'] = bboxes[1]
            left_lung['area'] = areas[1]
            right_lung['bbox'] = bboxes[0]
            right_lung['area'] = areas[0]
        else:
            left_lung['bbox'] = bboxes[0]
            left_lung['area'] = areas[0]
            right_lung['bbox'] = bboxes[1]
            right_lung['area'] = areas[1]
        
        left_lung['center'] = (int((left_lung['bbox'][0] + left_lung['bbox'][2])/2),
                               int((left_lung['bbox'][1] + left_lung['bbox'][3])/2))
        
        right_lung['center'] = (int((right_lung['bbox'][0] + right_lung['bbox'][2])/2),
                                int((right_lung['bbox'][1] + right_lung['bbox'][3])/2))

        infos["left_lung"] = left_lung
        infos["right_lung"] = right_lung

        #print('left:', left_lung)
        #print('right:', right_lung)
        #image = cv2.circle(image, left_lung['center'], radius=0, color=(255, 0, 0), thickness=10)
        #image = cv2.circle(image, right_lung['center'], radius=0, color=(0, 255, 0), thickness=10)
        #cv2.rectangle(image, (left_lung['bbox'][0], left_lung['bbox'][1]),
        #                     (left_lung['bbox'][2], left_lung['bbox'][3]), (255, 0, 0), 2)
        #
        #cv2.rectangle(image, (right_lung['bbox'][0], right_lung['bbox'][1]),
        #                     (right_lung['bbox'][2], right_lung['bbox'][3]), (0, 255, 0), 2)
        #cv2.imwrite('image.jpg', image)
    
        #print(json_name)
        with open(f'{json_path}/{json_name}', "w") as outfile:
            json.dump(infos, outfile, indent=2)
    #break

#mode = 'dataset'
#mode = 'stargan'
#mode = 'stylegan'

mode = sys.argv[1]

images_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/images/'
masks_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/lungs/'
json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{mode}/lung_areas/'

if os.path.isdir(json_path):
    os.system(f'rm -rf {json_path}')
os.mkdir(json_path)

images_path = glob.glob(f'{images_path}/*.jpg')
masks_path = glob.glob(f'{masks_path}/*.png')

images_path = sorted(images_path)
masks_path = sorted(masks_path)

#find_lung_areas(images_path[1], masks_path[1])

Parallel(n_jobs=-1, verbose=10)(delayed(find_lung_areas)(image_path, mask_path) for image_path, mask_path in zip(images_path, masks_path))
print('--- END ---')