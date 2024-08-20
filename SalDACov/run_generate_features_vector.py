import os

runs = [
    'python generate_features_vector.py dataset',
    'python generate_features_vector.py stargan',
    'python generate_features_vector.py stylegan',
]

for run in runs:
    os.system(run)