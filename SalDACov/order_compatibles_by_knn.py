import os
import sys
import cv2
import glob
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

def order(lung_features, analised_lung, compatibles_jsons):

    compatibles_features = []
    for json_file in compatibles_jsons:
        with open(json_file) as f:
            compatible = json.load(f)
        compatibles_features.append(compatible[analised_lung]['features'])

    knn = NearestNeighbors(n_neighbors=len(compatibles_features)).fit(compatibles_features)
    knn_result = knn.kneighbors([lung_features], len(compatibles_features), return_distance=False)
    knn_indexes = knn_result[0].tolist()

    #print(knn_result)
    #print(knn_indexes)

    return [compatibles_jsons[i] for i in knn_indexes]
    #for item in compatibles_ordered:
    #    print(item)

def order_by_knn(lung_info):

    with open(lung_info) as f:
        infos = json.load(f)

    infos['left_lung']['compatibles_left'] = order(infos['left_lung']['features'], 'left_lung', infos['left_lung']['compatibles_left'])
    infos['left_lung']['compatibles_right'] = order(infos['left_lung']['features'], 'right_lung', infos['left_lung']['compatibles_right'])

    infos['right_lung']['compatibles_right'] = order(infos['right_lung']['features'], 'right_lung', infos['right_lung']['compatibles_right'])
    infos['right_lung']['compatibles_left'] = order(infos['right_lung']['features'], 'left_lung', infos['right_lung']['compatibles_left'])
          
    json_name = lung_info.split('/')[-1]
    print(json_name)
    with open(f'{knn_json_path}/{json_name}', "w") as outfile:
        json.dump(infos, outfile, indent=2) 
    
gan = sys.argv[1]

json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/size_compatibles/'
knn_json_path = f'/home/bakrinski/nobackup/augmentation/augmentation/{gan}/ordered_by_knn/'

if os.path.isdir(knn_json_path):
    os.system(f'rm -rf {knn_json_path}')
os.mkdir(knn_json_path)

lung_infos = glob.glob(f'{json_path}/*.json')

#order_by_knn(lung_infos[0])
#for lung_info in lung_infos:
#    order_by_knn(lung_info)
#    break

Parallel(n_jobs=-1, verbose=10)(delayed(order_by_knn)(lung_info) for lung_info in lung_infos)
print('--- END ---')