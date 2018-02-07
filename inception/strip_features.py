# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:24:57 2018

@author: lbechberger
"""

import pickle, sys

source_file_name = sys.argv[1]
target_file_name = sys.argv[2]

source = pickle.load(open(source_file_name, 'rb'))

result = {}

for img_name in source.keys():
    augmented, _, _ = source[img_name]
    result[img_name] = augmented

pickle.dump(result, open(target_file_name, 'wb'))