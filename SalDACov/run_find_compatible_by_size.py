import os

runs = [
    'python find_compatible_by_size.py stargan',
    'python find_compatible_by_size.py stylegan',
]

for run in runs:
    os.system(run)