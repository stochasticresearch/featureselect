#!/usr/bin/env python

import os
import glob

import pandas as pd

import sys
sys.path.append(os.path.join(os.environ['HOME'],'gordian','metrics'))
import ml_lib

if __name__=='__main__':
    csvFile = '/data/rf_fingerprinting/data.csv'
    df = pd.read_csv(csvFile)

    mean_acc,std_acc,model_fname,metrics_fname = ml_lib.compute_accuracy(df, model_persist_dir='/tmp', 
                     label_column_name='transmitter_id', col_exclude_list=None,
                     num_gp_generations=50, num_classifiers_to_try=500)