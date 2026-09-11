
import pandas as pd
import numpy as np
import sys
import re
sys.path.append('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/best_models')
from utils.utils import *

reload_popt = False
reload_class_features = True

# set up
path = '/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv'

# load feature data
df_master = pd.read_csv(path,encoding='latin_1',low_memory=False)
df_static = pd.read_csv('data/static_22_03_24.csv')
df_phenotype_mapper = pd.read_csv('data/phenotype_mapper.csv')

# add static features
static_cols = [k for k in df_static.keys() if not k in df_master.keys()]
df_master = df_master.merge(df_static.loc[:,['id']+static_cols].rename(columns={'id':'id_old'}),on='id_old',how='left')

# features km
features_km(df_master)

# get initial features
df_features,dfcens = get_initial_features(df_master,df_phenotype_mapper)

# save the original features
df_features0 = df_features.copy()

# number of points
suffix = '83'
baseline_vals = {'q3':4, 'bulbar_subscore':12,  'ALSFRS_Total':48,  'Weight':np.nan,'% predicted':100}
for n_points in [2,4,6]:
    print('Computing number of points:',n_points,'...')
    A = []
    for var in baseline_vals.keys():
        cols = [col for col in df_features0.keys() if col.find(var)!=-1]
        # df_features0.loc[:,cols[0]] = df_features0.loc[:,cols[0]].fillna(baseline_vals[var])
        aux = df_features0.loc[:,cols]
        aux_notna = aux.notna()
        newaux = pd.concat([pd.DataFrame(aux.iloc[ri,np.where(row)[0][:n_points]],columns=[ri]).T for ri,row in aux_notna.iterrows()])
        cols_in_both = [c for c in aux.columns if c in newaux.columns]
        newaux = newaux.loc[:,cols_in_both]
        A.append(newaux)
    df_features = pd.concat([df_features0.loc[:,['id','Phenotype']]]+A+[df_features0.iloc[:,-14:]],axis=1)

    # df_features,dfcens = get_initial_features(df_master,df_phenotype_mapper)

    # get the class features
    df_class_feats = get_class_features_gt()
    df_class_feats2 = get_class_features_frechet(df_class_feats,df_features,reload_class_features=reload_class_features)

    # add ordinal classes
    df_class_feats = add_ordinal(df_class_feats,ignore_weight=False)
    df_class_feats2 = add_ordinal(df_class_feats2,ignore_weight=True)
    newcols = [c for c in df_class_feats.keys() if c not in list(df_class_feats2.keys())]
    df_features = df_class_feats2.merge(df_class_feats.loc[:,['id']+newcols],on='id').merge(df_features,on='id')

    # add longitudinal features
    # get all the of the first timepoint slopes
    df_features = df_features.rename(columns={'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS':'Slope_Onset_to_FirstBulbar',
                                              'ALSFRS_Slope_Onset_to_FirstALSFRS':'Slope_Onset_to_FirstALSFRS'})# renaming to keep consistency
    
    # % FEV 
    cols_fev = [c for c in df_features.columns if "_% predicted" in c and c.startswith('d')]
    df_features.loc[:,r'Slope_Onset_to_First%FEV'] = (100-df_features.loc[:,cols_fev].fillna(method='bfill', axis=1).iloc[:, 0])/(df_features.loc[:,cols_fev].notna().idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

    # ALSFRS_Slope_Onset_to_FirstWeight
    cols_weight = [c for c in df_features.columns if "Weight" in c and c.startswith('d')]
    df_features.loc[:,r'Slope_Onset_to_FirstWeight'] = (4-df_features.loc[:,cols_weight].fillna(method='bfill', axis=1).iloc[:, 0])/(df_features.loc[:,cols_weight].notna().idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

    # ALSFRS_Slope_Onset_to_Firstq3
    cols_q3 = [c for c in df_features.columns if "q3" in c and c.startswith('d')]
    df_features.loc[:,r'Slope_Onset_to_Firstq3'] = (4-df_features.loc[:,cols_q3].fillna(method='bfill', axis=1).iloc[:, 0])/(df_features.loc[:,cols_q3].notna().idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

    df_features.to_csv(f'data/allfeatures{suffix}_{n_points}.csv',index=False)

    print('Number of points:',n_points, 'Computed!!')


