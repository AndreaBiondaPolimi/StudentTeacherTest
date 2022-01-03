'''
Script to init a CSV file from MVTec dataset.
'''

import csv
import os
import sys


#dataset = sys.argv[1]
dataset = 'wood'
path = 'D:\Projects\student-teacher-anomaly-detection-master'

with open(path + f'\\data\\{dataset}\\{dataset}.csv', 'w') as csvfile:
    fieldnames = ['image_name', 'gt_name', 'label', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(path + f'\\data\\{dataset}/img'):
        fname, fext = os.path.splitext(filename)
        fname = fname.split('_')
        datatype = fname[0]
        img_id = fname[-1]
        anomaly = '_'.join(fname[1:-1])
        label = 0 if anomaly == 'good' else 1

        if label:
            gt = f'ground_truth_{anomaly}_{img_id}_mask' + fext
        else:
            gt = ''

        row = {'image_name': filename,
               'gt_name': gt,
               'label': label,
               'type': datatype}

        writer.writerow(row)

