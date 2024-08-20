import os

runs = [
    'python order_compatibles_by_knn.py stargan',
    'python order_compatibles_by_knn.py stylegan',
]

for run in runs:
    os.system(run)