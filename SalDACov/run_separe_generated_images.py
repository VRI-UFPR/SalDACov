import os

runs = [
    'python separe_generated_images.py stargan',
    'python separe_generated_images.py stylegan',
]

for run in runs:
    os.system(run)