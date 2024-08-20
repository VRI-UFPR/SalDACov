import os
import sys
import cv2
import glob
import random
import numpy as np

gan = sys.argv[1]

augmented_images_root = 'augmented_images_isis_ricord'

images_path = f'../{gan}/{augmented_images_root}/images/'
masks_path = f'../{gan}/{augmented_images_root}/masks/'
masked_path = f'../{gan}/{augmented_images_root}/masked/'

#augmentation = 294
#augmentation = 587
#augmentation = 880
#augmentation = 1174
#29774
#augmentation = 1489
#augmentation = 2978
#augmentation = 4467
#augmentation = 5955
#augmentation = 7444
#augmentation = 8933
#augmentation = 10421
#augmentation = 11910
#augmentation = 13399
#augmentation = 14888

#augmentations = [1489, 2978, 4467, 5955, 7444, 8933, 10421, 11910, 13399, 14888]
augmentations = [294, 587, 880, 1174, 1467, 1760, 2054, 2347, 2640, 2934]
ps = ['005', '010', '015', '020', '025', '030', '035', '040', '045', '050']

for p, augmentation in zip(ps, augmentations):
    print(p)
    print(augmentation)

    images = glob.glob(images_path + '*.jpg')
    random.shuffle(images)
    
    p_path = f'../{gan}/{augmented_images_root}/{p}/'
    if os.path.isdir(p_path):
        os.system('rm -rf {}'.format(p_path))
    os.mkdir(p_path)

    init = 0
    for i in range(5):
        
        augmented_images = p_path + 'augmented_images_fold' + str(i) + '/'
        augmented_masks = p_path + 'augmented_masks_fold' + str(i) + '/'
        augmented_masked = p_path + 'augmented_masked_fold'  + str(i) + '/'

        if os.path.isdir(augmented_images):
            os.system('rm -rf {}'.format(augmented_images))
        os.mkdir(augmented_images)

        if os.path.isdir(augmented_masks):
            os.system('rm -rf {}'.format(augmented_masks))
        os.mkdir(augmented_masks)

        if os.path.isdir(augmented_masked):
            os.system('rm -rf {}'.format(augmented_masked))
        os.mkdir(augmented_masked)

        #print(len(images))
        areas = []
        cont = 0
        for e, image in enumerate(images[init:]):
        
            cp_image = 'cp ' + image + ' ' + augmented_images
            cp_masked = 'cp ' + image.replace(images_path, masked_path) + ' ' + augmented_masked
            cp_masks = 'cp ' + image.replace(images_path, masks_path).replace('.jpg','.png') + ' ' + augmented_masks
            os.system(cp_image)
            os.system(cp_masks)
            os.system(cp_masked)
            cont += 1
            if cont == augmentation:
                init = e * (i + 1) + 1
                break
            '''
            mask_path = image.replace(images_path, masks_path).replace('.jpg','.png')
            mask = cv2.imread(mask_path,0)
            mask = np.where(mask != 0, 1, 0)
            area = np.sum(mask == 1)
            if area >= 5000:
                print(image.split('/')[-1])
                new_mask_path = mask_path.replace(masks_path, augmented_masks)
                cv2.imwrite(new_mask_path, mask)
                os.system(cp_image)
                os.system(cp_masked)
                cont += 1
            '''