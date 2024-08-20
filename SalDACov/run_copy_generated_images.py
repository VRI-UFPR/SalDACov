import os

runs = [
    'python copy_generated_images.py stargan',
    'python copy_generated_images.py stylegan',
]

for run in runs:
    os.system(run)