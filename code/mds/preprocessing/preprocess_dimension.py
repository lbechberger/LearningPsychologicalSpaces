# -*- coding: utf-8 -*-
"""
Read in the ratings about a given dimension and store it as pickle file.

Created on Thu Jan  9 11:48:19 2020

@author: lbechberger
"""

import pickle, argparse, csv

parser = argparse.ArgumentParser(description='Preprocessing dimension ratings')
parser.add_argument('binary_file', help = 'CSV file containing the binary dimension ratings')
parser.add_argument('continuous_file', help = 'CSV file containing the continuous dimension ratings')
parser.add_argument('output_file', help = 'path to the output pickle file')
parser.add_argument('-p', '--plot', action = 'store_true', help = 'plot histograms for the given values')
parser.add_argument('-o', '--output_folder', help = 'folder where the histograms shall be stored', default = '.')
args = parser.parse_args()

response_mapping = {'länglich': False, 'gleich': True, 'keineAhnung': None,
                    'gebogen': True, 'gerade': False}

item_name_to_id = {}
output = {}

rts = []
continuous = []

# read in information from binary ratings
with open(args.binary_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:
        picture_id = row['picture_id']
        item_id = picture_id.split('_')[0]
        
        if item_id not in output:
            item_name = row['item']
            # if not: add item information to dictionary
            output[item_id] = {'name': item_name, 'binary': [], 'continuous': []}
            item_name_to_id[item_name] = item_id
        
        rt = int(row['RT'])
        response = response_mapping[row['Response']]
        output[item_id]['binary'].append((rt, response))
        
        rts.append(rt)
        

# read in information from continuous ratings
with open(args.continuous_file, 'r') as f_in:
    
    reader = csv.DictReader(f_in, delimiter=',')
    for row in reader:

        for item_name, item_id in item_name_to_id.items():
            value = row[item_name]
            if len(value) > 0:
                # ignore empty entries
                output[item_id]['continuous'].append(int(value))
                continuous.append(int(value))

with open(args.output_file, 'wb') as f_out:
    pickle.dump(output, f_out)

# do the plotting
if args.plot:
    from matplotlib import pyplot as plt
    import os
    import numpy as np
    
    inverted_rts = -np.log(rts)

    for data, title in [(rts,'Response Time'), (continuous, 'Continuous Rating'), (inverted_rts, 'Negative Log RT')]:
        
        plt.hist(data, bins=21)
        plt.title('Histogram of {0}'.format(title))
        plt.xlabel(title)
        plt.ylabel('Number of Occurences')
        plot_output_file = os.path.join(args.output_folder, "{0}-hist.png".format(title))
        plt.savefig(plot_output_file, bbox_inches='tight', dpi=200)
        plt.close()