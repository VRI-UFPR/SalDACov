import os

runs = [
    'python run_find_lung_areas.py',
    'python run_generate_features_vector.py',
    'bash find_compatible_by_size.sh',
    #'bash order_compatibles_by_knn.sh',
]

for run in runs:
    os.system(run)