import os

runs = [
    'python generate_new_images.py stargan mxi mix mxi mix',
    'python generate_new_images.py stylegan mxi mix mxi mix',
]

for run in runs:
    os.system(run)