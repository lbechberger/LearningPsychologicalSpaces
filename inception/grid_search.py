# -*- coding: utf-8 -*-
"""
Creates configurations for a systematic grid search of hyperparameters.

Created on Wed Feb  7 15:31:59 2018

@author: lbechberger
"""

import sys
from configparser import ConfigParser


template_file_name = sys.argv[1]
input_config = ConfigParser()
input_config.read(template_file_name)
template_dict = dict(input_config['default'])

output_config = ConfigParser()

candidates_num_steps = [200, 500, 1000, 2000, 5000]
candidates_batch_size = [32, 64, 128, 256]
candidates_keep_prob = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
candidates_alpha = [0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0]
candidates_learning_rate = [0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020]

counter = 0
config_names = []
for num_steps in candidates_num_steps:
    for batch_size in candidates_batch_size:
        for keep_prob in candidates_keep_prob:
            for alpha in candidates_alpha:
                for learning_rate in candidates_learning_rate:
                    config_string = "n{0}-b{1}-k{2}-a{3}-l{4}".format(num_steps, batch_size, keep_prob, alpha, learning_rate)
                    config_dict = template_dict.copy()
                    local_dict = {'num_steps':num_steps, 'batch_size':batch_size, 'keep_prob':keep_prob, 
                                  'alpha':alpha, 'learning_rate':learning_rate}   
                    config_dict.update(local_dict)
                    output_config[config_string] = config_dict
                    config_names.append(config_string)
                    counter += 1

print("Created {0} configurations".format(counter))

with open('regression.cfg', 'w') as f:
    output_config.write(f)
    
with open('params.txt', 'w') as f:
    for config_name in config_names:
        f.write(config_name + '\n')