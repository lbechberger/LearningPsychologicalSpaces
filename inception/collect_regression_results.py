# -*- coding: utf-8 -*-
"""
Collects all regression results and stores them in a single csv file.

Created on Wed Feb  7 08:24:49 2018

@author: lbechberger
"""

import os, sys
folder_name = sys.argv[1]

with open(os.path.join(folder_name, "summary.csv"), 'w') as out_file:
    out_file.write("config,train_rmse,test_rmse\n")
    file_names = [fn for fn in os.listdir(folder_name) if not fn.endswith("csv")]
    for file_name in file_names:
        mean_train = 0
        mean_test = 0
        counter = 0
        with open(os.path.join(folder_name, file_name), 'r') as in_file:
            for line in in_file:
                if len(line) > 0:
                    [train, test] = line.split(',')
                    counter += 1
                    mean_train += float(train)
                    mean_test += float(test)
        mean_train = (1.0 * mean_train) / counter
        mean_test = (1.0 * mean_test) / counter
        out_file.write("{0},{1},{2}\n".format(file_name, mean_train, mean_test))
            