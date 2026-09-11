# hyperpar_opt_right_censored.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file allows to find best hyperparameters

from multiprocessing import Pool
from itertools import combinations, product
from datetime import datetime
from utils.utils_censor import *
import logging
import os

# Basic configuration
logging.basicConfig(filename=f'logs/Surv_{datetime.now().strftime("%Y%m%d_%H%M%S")}_train.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

_suffix = "111_margin_aj_1080_2920_mice"

if __name__ == '__main__':
    computed_methods = [] if ignore_computed_methods else [(l.split('_')[0],l.split('_')[1]) for l in os.listdir('logs') if l.endswith('.csv')] 

    # set combinations of models
    input_data = ((d, m) + load_data(d,m) + (logger,seed,n_trials) for d, m in product(datasets,methods) if not (d,m) in computed_methods)

    if paralell:
        # compute all data
        with Pool(6) as p:
            results = p.starmap(hyperopt_all_methods, input_data)
    else:
        results = []
        for pars in list(input_data):
            print(pars[0],pars[1])
            results.append(hyperopt_all_methods(*pars))

    # export all
    results = pd.concat(results,axis=0)

    results = order_features(results)
    
    # results2 = compute_best(results,vars = ['Cindex_ipcw','MedianAEPO_test'],mode='trial')
    results.to_csv(f'data/results_strat_xgbmaepo_weighted_all{_suffix}.csv',index=False)

    # get a summary and calculate pvalues
    best_results = results.query('is_best_trial_overall== True')
    best_results = compute_best(best_results,mode='rotation',vars = ['Cindex_ipcw','MedianAEPO_test'])
    # get best overall
    best_results = rank_and_score_weighted(best_results,'overall',[],vars = ['Cindex_ipcw','MedianAEPO_test'])[0]
    best_results.sort_values(by='overall_overall_score',inplace=True,ascending=False)

    best_results.to_csv(f'data/results_best_strat_xgbmaepo_weighted_all{_suffix}.csv',index=False)
    logger.info(f"Results saved to data/results_strat_xgbmaepo_weighted_all{_suffix}.csv")
    logger.info(f"Best results saved to data/results_best_strat_xgbmaepo_weighted_all{_suffix}.csv")
