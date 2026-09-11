# individual_feature_test.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file trains the model with individual features only.

import pandas as pd
import os
from utils.utils_censor import *
import logging
from datetime import datetime
from multiprocessing import Pool

reload = True
suffix = '95'

# Basic configuration
logging.basicConfig(filename=f'logs/Surv_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pred.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    models_df = pd.read_csv('data/results_best_strat_xgbmaepo_weighted_all111_margin_aj_final1080_2920_2.csv')
    # models_df.groupby(['test_fold','method','dataset'])['Cindex_test_uncensored'].describe()
    models_df = models_df.query('is_best_overall_overall == True')
    # models_df['dataset'] = "classes_slope"

    # create keys 
    models_df['key'] = models_df['dataset']+models_df['method']

    # models_df = models_df.sort_values(by=['best_trial']).drop_duplicates(subset=['dataset', 'method'])
    models_df['dataset'] = 'indiv'
    
    # load data
    input_data = (tuple([row])+load_data(row.dataset,row.method) + tuple([logger]) for ri,row in models_df.iterrows())

    # call the model
    res = []
    for pars in input_data:
        res.append(train_all_methods_wrapper(*pars))
    
    # decode the results
    results,summary,dfpred = [],[],[]
    for r in res:
        summary.append(r[1])
        results.append(r[0])
        dfpred.append(r[2])
    results = pd.concat(results,axis=0,ignore_index=True)
    summary = pd.concat(summary,axis=0,ignore_index=True)
    dfpred = pd.concat(dfpred,axis=0,ignore_index=True)
    results.to_csv(f'data/acc_curve_simulation_last0_5Final_indiv{suffix}.csv',index=False)
    summary.to_csv(f'data/summary_database_rotation_best_last0_5Final_indiv{suffix}.csv',index=False)
    dfpred.to_csv(f'data/predictions_last0_5Final_indiv{suffix}.csv',index=False)

    summary_sum = summary.loc[:,['method',	'dataset',
                    'MedianAE_test','Cindex_test']]
    summary_sum = summary_sum.groupby(['method','dataset']).mean().reset_index()

    # minmax = MinMaxScaler()
    # summary_sum['overall_score'] = 1/3*(summary_sum['Acc90_test_uncensored'].values.reshape(-1,1)+summary_sum['Cindex_test_uncensored'].values.reshape(-1,1)+1-minmax.fit_transform(summary_sum['MedianAE_test_uncensored'].values.reshape(-1,1)))
    # summary_sum.sort_values(by=['overall_score'],ascending=False,inplace=True)
    summary_sum.to_csv(f'data/summary_database_rotation_best_last0_5Final_avg_indiv{suffix}.csv',index=False)
