import os
import sys
import cv2
import glob
import json
import random
import numpy as np
import albumentations
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

def add_lesion(infos, image, new_lesion_mask, lung_mask, compatible_lung_json, lung, flip):

    lim_area = 100

    transform = albumentations.HorizontalFlip(always_apply=True)
    
    with open(compatible_lung_json) as g:
        compatible_lung = json.load(g)

    print(compatible_lung['image_path'])

    compatible_image = cv2.imread(compatible_lung['image_path'])
    compatible_lung_mask = cv2.imread(compatible_lung['lung_mask_path'], 0)
    compatible_lesion_mask = cv2.imread(compatible_lung['lesion_mask_path'], 0)
    height, width = compatible_lesion_mask.shape

    compatible_lung_mask = np.where(compatible_lung_mask != 0, 255, 0)
    compatible_lesion_mask = np.where(compatible_lesion_mask != 0, 255, 0)

    #cv2.imwrite(f'lesion_image_{lung}.jpg', compatible_image)
    #cv2.imwrite(f'lesion_mask_{lung}.jpg', compatible_lesion_mask)
    #area = int(np.sum(compatible_lesion_mask == 255))
    #print(area)
    #if area < lim_area:
    #    return

    #flip = False
    if flip:
        compatible_image = transform(image=compatible_image)['image']
        compatible_lung_mask = transform(image=compatible_lung_mask)['image']
        compatible_lesion_mask = transform(image=compatible_lesion_mask)['image']
        
        if lung == 'left_lung':
            flung = 'right_lung'
        else:
            flung = 'left_lung'

        aux = width - compatible_lung[flung]['bbox'][2]
        compatible_lung[flung]['bbox'][2] = width - compatible_lung[flung]['bbox'][0]
        compatible_lung[flung]['bbox'][0] = aux

        compatible_lung[flung]['center'] = (int((compatible_lung[flung]['bbox'][0] + compatible_lung[flung]['bbox'][2])/2),
                                           int((compatible_lung[flung]['bbox'][1] + compatible_lung[flung]['bbox'][3])/2))
        
        compatible_lung[lung]['bbox'] = compatible_lung[flung]['bbox']
        compatible_lung[lung]['center'] = compatible_lung[flung]['center']

        #cv2.imwrite(f'lesion_image_{lung}_flipped.jpg', compatible_image)
        #cv2.imwrite(f'lesion_mask_{lung}_flipped.jpg', compatible_lesion_mask)
    
    #compatible_lung_mask = compatible_lung_mask.astype(np.uint8)
    #compatible_lesion_mask1 = compatible_lesion_mask.astype(np.uint8)
    #zeros = np.zeros((height, width), np.uint8)
    #compatible_lung_mask = cv2.merge([zeros, compatible_lung_mask,zeros])
    #compatible_lesion_mask1 = cv2.merge([zeros, compatible_lesion_mask1,zeros])
    #compatible_image = cv2.addWeighted(compatible_image, 1, compatible_lung_mask, 1,0)
    
    #cv2.rectangle(compatible_image, (compatible_lung[lung]['bbox'][0], compatible_lung[lung]['bbox'][1]),
    #                                (compatible_lung[lung]['bbox'][2], compatible_lung[lung]['bbox'][3]), (255, 0, 0), 2)
    
    #cv2.imwrite(f'lesion_lesion_{lung}.jpg', compatible_image)

    for y in range(height):
        for x in range(width):
            if (x <= compatible_lung[lung]['bbox'][0] or x >= compatible_lung[lung]['bbox'][2]) or \
                (y <= compatible_lung[lung]['bbox'][1] or y >= compatible_lung[lung]['bbox'][3]):
                    compatible_image[y,x] = 0
                    compatible_lesion_mask[y,x] = 0
    
    #cv2.imwrite('lesion_image_cropped.jpg', compatible_image)
    #cv2.imwrite(f'lesion_mask_{lung}_cropped.jpg', compatible_lesion_mask)
    area = int(np.sum(compatible_lesion_mask == 255))
    if area < lim_area:
        return

    shift_x = infos[lung]['center'][0] - compatible_lung[lung]['center'][0]
    shift_y = infos[lung]['center'][1] - compatible_lung[lung]['center'][1]
    translation_matrix = np.float32([[1,0,shift_x], [0,1,shift_y]])

    compatible_image = cv2.warpAffine(compatible_image, translation_matrix, (height, width))
    
    compatible_lesion_mask = compatible_lesion_mask.astype(np.float32)
    compatible_lesion_mask = cv2.warpAffine(compatible_lesion_mask, translation_matrix, (height, width))

    #cv2.imwrite(f'lesion_image_{lung}_shifted.jpg', compatible_image)
    #cv2.imwrite(f'lesion_mask_{lung}_shifted.jpg', compatible_lesion_mask)
    #area = int(np.sum(compatible_lesion_mask == 255))
    #print(area)
    #if area < lim_area:
    #    return
    
    for y in range(height):
        for x in range(width):
            if lung_mask[y,x] == 0 and compatible_lesion_mask[y,x] != 0:
                compatible_lesion_mask[y,x] = 0
    
    #cv2.imwrite(f'lesion_mask_{lung}_lunged.jpg', compatible_lesion_mask)
    area = int(np.sum(compatible_lesion_mask == 255))
    if area < lim_area:
        return
    #image = image.astype(np.uint8)
    compatible_lesion_mask = compatible_lesion_mask.astype(np.uint8)

    contours, hiers = cv2.findContours(compatible_lesion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    minx = width
    miny = height
    maxx = 0
    maxy = 0
    for contour in contours:
        x1, y1, w, h = cv2.boundingRect(contour)
        x2 = x1 + w
        y2 = y1 + h
        if x1 < minx:
            minx = x1
        if y1 < miny:
            miny = y1
        if x2 > maxx:
            maxx = x2
        if y2 > maxy:
            maxy = y2
        
    #x, y, w, h = cv2.boundingRect(contours[0])
    center = (int((minx + maxx)/2), int((miny + maxy)/2))
        
    #output = cv2.seamlessClone(compatible_image, 
    #                           image,                                
    #                           compatible_lesion_mask, 
    #                           center, cv2.NORMAL_CLONE)
        
    new_lesion_mask = cv2.addWeighted(new_lesion_mask, 1, compatible_lesion_mask, 1,0)
    
    m = np.where(new_lesion_mask != 0, 1, 0)
    #zeros = np.zeros((height, width), np.uint8)
    m = cv2.merge([m, m, m])
    
    compatible_image = compatible_image * m
    
    image = image.astype(np.uint8)
    compatible_image = compatible_image.astype(np.uint8)
    
    output = cv2.addWeighted(image, 1, compatible_image, 1, 0)
    
    return output, new_lesion_mask

def generate_new_images(lung_info):

    print(lung_info)

    with open(lung_info) as f:
        infos = json.load(f)

    image_name = infos['image_path'].split('/')[-1]

    image = cv2.imread(infos['image_path'])
    lung_mask = cv2.imread(infos['lung_mask_path'], 0)

    #cv2.imwrite('gan_image.jpg',image)
    lung_mask = np.where(lung_mask != 0, 255, 0)
    #cv2.imwrite('gan_mask.png', lung_mask)

    height, width, _ = image.shape

    new_lesion_mask = np.zeros((height, width), np.uint8)

    # LEFT LUNG
    #--------------------------------------------------
    flip = False
    if lung_position_left == 'same':
        #print('same')
        compatible_lungs_left = infos['left_lung']['compatibles_left']
    elif lung_position_left == 'flipped':
        #print('flipped')
        flip = True
        compatible_lungs_left = infos['left_lung']['compatibles_right']
    elif lung_position_left == 'mix':
        if random.randint(0, 1):
            compatible_lungs_left = infos['left_lung']['compatibles_left']
        else:
            flip = True
            compatible_lungs_left = infos['left_lung']['compatibles_right']
    
    if saliency_mode_left == 'min':
        compatible_lung_json_left = compatible_lungs_left[random.randint(0, 9)]
    elif saliency_mode_left == 'max':
        compatible_lung_json_left = compatible_lungs_left[random.randint(len(compatible_lungs_left)-10, len(compatible_lungs_left)-1)]
    elif saliency_mode_left == 'mxi':
        compatible_lung_json_left = compatible_lungs_left[random.randint(0, len(compatible_lungs_left)-1)]

    r = add_lesion(infos, image, new_lesion_mask, lung_mask, compatible_lung_json_left, 'left_lung', flip)
    if r == None:
        return
    image, new_lesion_mask = r
        
    # RIGHT LUNG
    #--------------------------------------------------
    flip = False
    if lung_position_right == 'same':
        compatible_lungs_right = infos['right_lung']['compatibles_right']
    elif lung_position_right == 'flipped':
        flip = True
        compatible_lungs_right = infos['right_lung']['compatibles_left']
    elif lung_position_right == 'mix':
        if random.randint(0, 1):
            compatible_lungs_right = infos['right_lung']['compatibles_right']
        else:
            flip = True
            compatible_lungs_right = infos['right_lung']['compatibles_left']
    
    if saliency_mode_right == 'min':
        #compatible_lung_json_right = compatible_lungs_right[0]
        compatible_lung_json_right = compatible_lungs_right[random.randint(0, 9)]
    elif saliency_mode_right == 'max':
        #compatible_lung_json_right = compatible_lungs_right[len(compatible_lungs_right)-1]
        compatible_lung_json_right = compatible_lungs_right[random.randint(len(compatible_lungs_right)-10, len(compatible_lungs_right)-1)]
    elif saliency_mode_right == 'mxi':
        compatible_lung_json_right = compatible_lungs_right[random.randint(0, len(compatible_lungs_right)-1)]

    r = add_lesion(infos, image, new_lesion_mask, lung_mask, compatible_lung_json_right, 'right_lung', flip)
    if r == None:
        return
    image, new_lesion_mask = r
    #cv2.imwrite('gan_image_output_lesion_mask2.png', new_lesion_mask)
    #cv2.imwrite('gan_image_output.jpg', image)

    zeros = np.zeros((height, width), np.uint8)
    merged = cv2.merge([new_lesion_mask, zeros, zeros])

    masked_image = cv2.addWeighted(image, 1, merged, 1, 0)
    
    mask_name = image_name.replace('.jpg','.png')

    #cv2.imwrite('gan_image_new.jpg', image)
    cv2.imwrite(f'{new_images_path}/{image_name}', image)
    new_lesion_mask = np.where(new_lesion_mask != 0, 1, 0)
    cv2.imwrite(f'{new_masks_path}/{mask_name}', new_lesion_mask)
    cv2.imwrite(f'{new_masked_path}/{image_name}', masked_image)
    
    #cv2.imwrite('gan_image_output_lesion_mask1.jpg', masked_image)

gan = sys.argv[1]
saliency_mode_left = sys.argv[2]
lung_position_left = sys.argv[3]

saliency_mode_right = sys.argv[4]
lung_position_right = sys.argv[5]

augmented_images = f'augmented_images_{saliency_mode_left[1]}{lung_position_left[0]}{saliency_mode_right[1]}{lung_position_right[0]}'

json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/ordered_by_knn/'

augmented_images = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/{augmented_images}/'

if os.path.isdir(augmented_images):
    os.system(f'rm -rf {augmented_images}')
os.mkdir(augmented_images)

new_images_path = f'{augmented_images}/images'
new_masks_path = f'{augmented_images}/masks'
new_masked_path = f'{augmented_images}/masked'

os.mkdir(new_images_path)
os.mkdir(new_masks_path)
os.mkdir(new_masked_path)

lung_infos = glob.glob(f'{json_path}/*.json')

#for i, lung_info in enumerate(lung_infos): ####
#    print(f'index: {i}')
#    generate_new_images(lung_info)
#break
#generate_new_images(lung_infos[150])

Parallel(n_jobs=8)(delayed(generate_new_images)(lung_info) for lung_info in lung_infos)
print('--- END ---')