# -*- coding: utf-8 -*-
"""
Visualizes the correlations measured between the similarity ratings and the MDS/pixel-based ones.

Created on Wed Dec 12 09:01:26 2018

@author: lbechberger
"""

import argparse, os, csv
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Visualizing correlations')
parser.add_argument('pixel_file', help = 'the input file containing the results of the pixel-wise similarities')
parser.add_argument('mds_file', help = 'the input file containing the results of the MDS-based similarities')
parser.add_argument('-o', '--output_folder', help = 'the folder to which the output should be saved', default='.')
args = parser.parse_args()

pixel_data = {'pearson':{}, 'spearman':{}, 'kendall':{}, 'r2_linear':{}, 'r2_isotonic':{}}
mds_data = {'pearson':{}, 'spearman':{}, 'kendall':{}, 'r2_linear':{}, 'r2_isotonic':{}}

# read in pixel-based information
with open(args.pixel_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for metric, metric_dict in pixel_data.items():
            
            # scoring = Euclidean, Manhattan, Cosine, MutualInformation
            if row['scoring'] not in metric_dict.keys():
                metric_dict[row['scoring']] = {}
                
            # aggregator = min, max, mean, ...
            if row['aggregator'] not in metric_dict[row['scoring']].keys():
                metric_dict[row['scoring']][row['aggregator']] = []
            
            metric_dict[row['scoring']][row['aggregator']].append([int(row['block_size']), float(row[metric])])

# read in MDS-based information
with open(args.mds_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for metric, metric_dict in mds_data.items():
            
            # scoring = Euclidean, Manhattan, Cosine
            if row['scoring'] not in metric_dict.keys():
                metric_dict[row['scoring']] = []
            
            metric_dict[row['scoring']].append([int(row['n_dims']), float(row[metric])])


# now use that data to create some plots and some statistics
for metric in pixel_data.keys():

    pixel_dict = pixel_data[metric]
    mds_dict = mds_data[metric]
    
    best_pixel_result_metric = ['None', -1000]
    best_mds_result_metric = ['None', -1000]    
    
    for scoring in pixel_dict.keys():
        
        pixel_scores = pixel_dict[scoring]
        best_pixel_result_scoring = ['None', -1000]
        if scoring in mds_dict.keys():
            mds_scores = mds_dict[scoring]
            x_mds = []

        # first the plot for the pixel data...
        label_list = []
        fig, ax = plt.subplots(figsize=(24,12))
        for aggregator, values in pixel_scores.items():
            # sort and plot
            sorted_values = sorted(values, key = lambda x: x[0])
            bar_indices = np.arange(len(sorted_values))
            y_values = list(map(lambda x: x[1], sorted_values))
            legend = list(map(lambda x: x[0], sorted_values))
            label_list.append(aggregator)
            ax.plot(bar_indices, y_values)
            
            # checking whether this result was better
            max_val = max(map(lambda x: x[1], sorted_values))
            if max_val > best_pixel_result_scoring[1]:
                # find corresponding i
                max_i = 0
                for i, val in (sorted_values):
                    if val == max_val:
                        max_i = i
                        break
                best_pixel_result_scoring = ['{0}-{1}'.format(aggregator, max_i), max_val]

                
        # add best MDS
#        if scoring in mds_dict.keys():
#            y_mds = [max(map(lambda x: x[1], mds_scores))]*len(bar_indices)
#            label_list.append('best MDS')
#            ax.plot(bar_indices, y_mds)
            
        
        # create the final plot and store it
        ax.set_xlabel('Block Size', fontsize = 20)
        ax.set_xticks(bar_indices)
        ax.set_xticklabels(legend)
        ax.set_ylabel(metric, fontsize = 20)
        if metric == 'r2':
            ax.set_ylim(-1,1)
        plt.legend(label_list, fontsize = 20)
        output_file_name = os.path.join(args.output_folder, '{0}-{1}-pixel.jpg'.format(metric, scoring))
        plt.savefig(output_file_name, bbox_inches='tight', dpi=200)
        plt.close()
    
        # print best pixel result
        print('best pixel result for {0}-{1}: {2}'.format(metric, scoring, best_pixel_result_scoring))
        if best_pixel_result_scoring[1] > best_pixel_result_metric[1]:
            best_pixel_result_metric = ['{0}-{1}'.format(scoring, best_pixel_result_scoring[0]), best_pixel_result_scoring[1]]
    
        # ... now the plot for the MDS data
        fig, ax = plt.subplots(figsize=(24,12))
        if scoring in mds_dict.keys():
            # sort, plot, and store
            sorted_values = sorted(mds_scores, key = lambda x: x[0])
            bar_indices = np.arange(len(sorted_values))
            legend = list(map(lambda x: x[0], sorted_values))
            y_values = list(map(lambda x: x[1], sorted_values))
            ax.plot(bar_indices, y_values)
            
            # add best pixel
            y_pixel = [best_pixel_result_metric[1]]*len(bar_indices)
            legend.append('best pixel: {0}'.format(best_pixel_result_metric[0]))
            ax.plot(bar_indices, y_pixel)

            ax.set_xlabel('Number of Dimensions', fontsize = 20)
            ax.set_ylabel(metric, fontsize = 20)
            ax.set_xticks(bar_indices)
            ax.set_xticklabels(legend)
            output_file_name = os.path.join(args.output_folder, '{0}-{1}-MDS.jpg'.format(metric, scoring))
            plt.savefig(output_file_name, bbox_inches='tight', dpi=200)
            plt.close()
            
            # look for best result
            max_val = max(map(lambda x: x[1], sorted_values))
            max_i = 0
            for i, val in (sorted_values):
                if val == max_val:
                    max_i = i
                    break
            print('best MDS result for {0}-{1}: {2}'.format(metric, scoring, [max_i, max_val]))
            if max_val > best_mds_result_metric[1]:
                best_mds_result_metric = ['{0}-{1}'.format(scoring, max_i), max_val]
    
    # print best results for given metric overall
    print('\t best pixel result for {0}: {1}'.format(metric, best_pixel_result_metric))
    print('\t best MDS result for {0}: {1}'.format(metric, best_mds_result_metric))
            