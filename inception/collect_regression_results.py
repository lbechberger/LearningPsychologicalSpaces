# -*- coding: utf-8 -*-
"""
Collects all regression results and stores them in a single csv file.

Created on Wed Feb  7 08:24:49 2018

@author: lbechberger
"""

import os

with open("regression/summary.csv", 'w') as out_file:
    out_file.write("config, train_rmse, test_rmse\n")
    file_names = [fn for fn in os.listdir("regression/") if not fn.endswith("csv")]
    for file_name in file_names:
        with open("regression/{0}".format(file_name), 'r') as in_file:
            for line in in_file:
                if len(line) > 0:
                    out_file.write("{0},{1}".format(file_name, line))
            