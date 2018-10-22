# -*- coding: utf-8 -*-
"""
Simply collects all individual files and dumps them into a file named 'all'

Created on Tue Feb  6 09:10:15 2018

@author: lbechberger
"""

import pickle
import sys, os

input_folder = sys.argv[1]
output_folder = sys.argv[2]

input_file_names = [f for f in os.listdir(input_folder) if f != 'all']

result = {}

for file_name in input_file_names:
    result[file_name] = pickle.load(open(os.path.join(input_folder, file_name), 'rb'))

pickle.dump(result, open(os.path.join(output_folder, 'images'), 'wb'))