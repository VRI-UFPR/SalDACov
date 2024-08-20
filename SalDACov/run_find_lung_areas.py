import os

runs = [
    'python find_lung_areas.py dataset',
    'python find_lung_areas.py stargan',
    'python find_lung_areas.py stylegan',
]

for run in runs:
    os.system(run)