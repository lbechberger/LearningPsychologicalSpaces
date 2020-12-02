# -*- coding: utf-8 -*-
"""
Count the number of valid sketches in Sketchy database.

Created on Wed Dec  2 10:25:18 2020

@author: lbechberger
"""


import argparse, csv

parser = argparse.ArgumentParser(description='Scan Sketchy labels')
parser.add_argument('input_file', help = 'csv file containing the Sketchy info')
args = parser.parse_args()

count_dict = {}

with open(args.input_file, 'r') as f_in:
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        cat = row['Category']
        if cat not in count_dict:
            count_dict[cat] = {'all':0, 'error':0, 'context':0, 'ambiguous':0, 'pose':0}
        count_dict[cat]['all'] += 1
        count_dict[cat]['error'] += int(row['Error?'])
        count_dict[cat]['context'] += int(row['Context?'])
        count_dict[cat]['ambiguous'] += int(row['Ambiguous?'])
        count_dict[cat]['pose'] += int(row['WrongPose?'])

cats = sorted(count_dict.keys())

print('category,#sketches,#error,#context,#ambiguous,#pose')
for cat in cats:
    print('{0},{1},{2},{3},{4},{5}'.format(cat, count_dict[cat]['all'], 
                                                  count_dict[cat]['error'], 
                                                    count_dict[cat]['context'], 
                                                    count_dict[cat]['ambiguous'], 
                                                    count_dict[cat]['pose']))