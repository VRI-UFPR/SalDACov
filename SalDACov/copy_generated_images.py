import os
import sys
import glob

gan = sys.argv[1]

dataset = 'total'
dataset_path = '../../total/train'

ps = ['005', '010', '015', '020', '025', '030', '035', '040', '045', '050']

aug_images = 'augmented_images_imim'

for p in ps:
    for i in range(5):

        augmented_images_path = f'../{gan}/{aug_images}/{p}/augmented_images_fold{i}/'
        augmented_masks_path = f'../{gan}/{aug_images}/{p}/augmented_masks_fold{i}/'

        #print(augmented_images_path)

        dataset_images = f'{dataset_path}/images_{gan}{p}_fold{i}/'
        dataset_masks = f'{dataset_path}/masks_{gan}{p}_fold{i}/'

        if os.path.isdir(dataset_images):
            os.system('rm -rf {}'.format(dataset_images))
        os.mkdir(dataset_images)

        if os.path.isdir(dataset_masks):
            os.system('rm -rf {}'.format(dataset_masks))
        os.mkdir(dataset_masks)

        augmented_images = glob.glob(f'{augmented_images_path}*.jpg')

        new_paths = []
        for augmented_image in augmented_images:
            #print(augmented_image)
            augmented_mask = augmented_image.replace(augmented_images_path, augmented_masks_path).replace('.jpg','.png')

            cp_image = 'cp ' + augmented_image + ' ' + dataset_images
            cp_mask = 'cp ' + augmented_mask + ' ' + dataset_masks

            os.system(cp_image)
            os.system(cp_mask)

            image_name = augmented_image.split('/')[-1]
            mask_name = augmented_mask.split('/')[-1]
            new_paths.append(f'datasets/{dataset}/train/images_{gan}{p}_fold{i}/{image_name} datasets/{dataset}/train/masks_{gan}{p}_fold{i}/{mask_name}')

        #for i in range(5):
        dataset_file_path = f'{dataset_path}/train_ids{i}.txt'

        with open(dataset_file_path, 'r') as dataset_file:
            dataset_paths = dataset_file.read().splitlines()

        all_paths = dataset_paths + new_paths

        new_dataset_file_path = f'{dataset_path}/train_ids{i}_{gan}{p}.txt'

        last = all_paths[-1]
        with open(new_dataset_file_path, 'w') as new_file:
            for pth in all_paths:
                if pth == last:
                    new_file.write(pth)
                else:
                    new_file.write(pth + '\n')
