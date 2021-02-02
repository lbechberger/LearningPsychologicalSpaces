# -*- coding: utf-8 -*-
"""
Averages the results per fold over all folds and stores them in a new csv file.

Created on Mon Feb  1 14:54:11 2021

@author: lbechberger
"""

import  argparse
from code.util import read_csv_results_files, write_csv_results_file

parser = argparse.ArgumentParser(description='Average fold results')
parser.add_argument('input_file', help = 'csv file containing the individual fold results')
parser.add_argument('output_file', help = 'csv file for the aggregated results')
args = parser.parse_args()

headline, content = read_csv_results_files(args.input_file, [''], ['configuration', 'fold'], 'configuration')
write_csv_results_file(args.output_file, headline, content, 'configuration')

