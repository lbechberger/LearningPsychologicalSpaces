# -*- coding: utf-8 -*-
"""
Shuffle the ground truth labels and store both the original and the shuffled form in a pickle file.

Created on Wed Feb  7 14:50:22 2018

@author: lbechberger
"""

import sys, random
import numpy as np
import pickle

random.seed(42)
source_file_name = sys.argv[1]
target_file_name = sys.argv[2]

real_dict = {}
shuffled_dict = {}

with open(source_file_name, 'r') as f:
    for line in f:
        if len(line) > 0:
            tokens = line.split(',')
            img_name = tokens[0]
            vector = tokens[1:]
            real_dict[img_name] = np.array(list(map(lambda x: float(x), vector)))

keys = list(real_dict.keys())
values = list(real_dict.values())
random.shuffle(values)
for i in range(len(keys)):
    shuffled_dict[keys[i]] = values[i]

result = {'targets':real_dict, 'shuffled':shuffled_dict}
pickle.dump(result, open(target_file_name, 'wb'))