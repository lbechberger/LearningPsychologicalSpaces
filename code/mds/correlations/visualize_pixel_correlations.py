# -*- coding: utf-8 -*-
"""
Visualizes the correlations measured between the dissimilarity ratings and pixel-based distances.

Created on Wed Dec 12 09:01:26 2018

@author: lbechberger
"""

import argparse, os, csv
from matplotlib import pyplot as plt
import numpy as np
from code.util import add_correlation_metrics_to_parser, get_correlation_metrics_from_args

parser = argparse.ArgumentParser(description='Visualizing correlations')
parser.add_argument('pixel_file', help = 'the input file containing the results of the pixel-wise similarities')
parser.add_argument('output_folder', help = 'the folder to which the output should be saved')
add_correlation_metrics_to_parser(parser)
args = parser.parse_args()

correlation_metrics = get_correlation_metrics_from_args(args)

pixel_data = {}
for metric in correlation_metrics:
    pixel_data[metric] = {}

metric_display = {'pearson': r"Pearson's $r$", 'spearman': r"Spearman's $\rho$", 'kendall': r"Kendall's $\tau$",
                      'r2_linear': r"$R^2$ based on a Linear Regression", 'r2_isotonic': r"$R^2$ based on a Isotonic Regression"}

# read in pixel-based information
with open(args.pixel_file, 'r') as f_in:
    reader = csv.DictReader(f_in)
    for row in reader:
        for metric, metric_dict in pixel_data.items():
            
            # scoring = Euclidean, Manhattan, inner product
            if row['scoring'] not in metric_dict.keys():
                metric_dict[row['scoring']] = {}
            
            if row['weights'] not in metric_dict[row['scoring']].keys():
                metric_dict[row['scoring']][row['weights']] = {}
            
            # aggregator = min, max, mean, ...
            if row['aggregator'] not in metric_dict[row['scoring']][row['weights']].keys():
                metric_dict[row['scoring']][row['weights']][row['aggregator']] = []
            
            metric_dict[row['scoring']][row['weights']][row['aggregator']].append([int(row['block_size']), float(row[metric])])


# now use that data to create some plots and some statistics
for metric in pixel_data.keys():

    pixel_dict = pixel_data[metric]
    
    best_pixel_result_metric = ['None', -1000]
    
    for scoring in pixel_dict.keys():
        
        for weights in pixel_dict[scoring].keys():        
        
            pixel_scores = pixel_dict[scoring][weights]
            best_pixel_result_scoring = ['None', -1000]
    
            # first the plot for the pixel data...
            label_list = []
            fig, ax = plt.subplots(figsize=(24,12))

            aggregators = zip(sorted(pixel_scores.keys()),['-', '--', '-.', ':']*(int(len(pixel_scores.keys())/4)))         
            
            for aggregator, line_style in aggregators:
                
                values = pixel_scores[aggregator]
                
                # sort and plot
                sorted_values = sorted(values, key = lambda x: x[0])
                bar_indices = np.arange(len(sorted_values))
                y_values = list(map(lambda x: x[1], sorted_values))
                legend = list(map(lambda x: x[0], sorted_values))
                label_list.append(aggregator)
                ax.plot(bar_indices, y_values, line_style, linewidth=3)
                
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
    
                    
            # create the final plot and store it
            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)
            ax.set_xlabel('Block Size', fontsize = 20)
            ax.set_xticks(bar_indices)
            ax.set_xticklabels(legend)
            ax.set_ylabel(metric_display[metric], fontsize = 20)
            if metric == 'r2':
                ax.set_ylim(-1,1)
            plt.legend(label_list, fontsize = 20)
            output_file_name = os.path.join(args.output_folder, '{0}-{1}-{2}-pixel.jpg'.format(metric, scoring, weights))
            plt.savefig(output_file_name, bbox_inches='tight', dpi=200)
            plt.close()
        
            # print best pixel result
            print('best pixel result for {0}-{1}-{2}: {3}'.format(metric, scoring, weights, best_pixel_result_scoring))
            if best_pixel_result_scoring[1] > best_pixel_result_metric[1]:
                best_pixel_result_metric = ['{0}-{1}-{2}'.format(scoring, weights, best_pixel_result_scoring[0]), best_pixel_result_scoring[1]]
            
    # print best results for given metric overall
    print('\t best pixel result for {0}: {1}'.format(metric, best_pixel_result_metric))
            