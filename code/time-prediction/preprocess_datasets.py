# preprocess_datasets.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file preprocesses datasets


import pandas as pd
import numpy as np
import sys
import re
sys.path.append('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/best_models')
from utils.utils import *

fulldatapath = "/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data"

reload_popt = True
reload_class_features = True

# load feature data
df_master = pd.read_csv(fulldatapath+'/master_final_0807.csv',encoding='latin_1',low_memory=False)
df_static = pd.read_csv(fulldatapath+'/static_22_03_24.csv')
df_phenotype_mapper = pd.read_csv(fulldatapath+'/phenotype_mapper.csv')

# add static features
static_cols = [k for k in df_static.keys() if not k in df_master.keys()]
df_master = df_master.merge(df_static.loc[:,['id']+static_cols].rename(columns={'id':'id_old'}),on='id_old',how='left')

# features km
features_km(df_master)

# get initial features
df_features,dfcens = get_initial_features(df_master,df_phenotype_mapper)

# classes data
# df_classes = pd.read_csv('data/Training_classes.csv') # this is for training classes only

# get the class features
df_class_feats = get_class_features_gt()
df_class_feats2 = get_class_features_frechet(df_class_feats,df_features,reload_class_features=reload_class_features)

# add ordinal classes
df_class_feats = add_ordinal(df_class_feats)
df_class_feats2 = add_ordinal(df_class_feats2,ignore_weight=True)

# export classes for evaluation
df_class_feats.merge(df_class_feats2,on='id',suffixes=['_gt','_est']).to_csv('data/classfeatures_frechet_and_gt.csv',index=False)

# add the frechet imputed classes
df_class_feats = df_class_feats.merge(df_class_feats2.loc[:,'id'],on='id',how='outer')
for c in df_class_feats.keys():
    if 'weight' in c:
        continue
    idx = df_class_feats[c].isna()
    ids = df_class_feats.loc[idx,'id'].values
    idx2 = df_class_feats2['id'].isin(ids)
    df_class_feats.loc[idx,c] = df_class_feats2.loc[idx2,c]

# reprocess raw features
df_features0 = df_features.copy()
X,y,ids,df_features = get_raw_feartures(df_features,dfcens)

# get decay features
dfdecay,y = get_decay_features(dfcens,df_features,ids,reload_popt=reload_popt)

# define encals features
encals_cols = ['sex_Female','site_onset_Spinal', 'age_at_onset', 'diagnostic_delay_months','ALSFRS_Slope_Onset_to_FirstALSFRS','mean_fev','C9orf72']

# merge all the features
decay_cols = [k for k in dfdecay.keys() if k not in df_features.keys()]
raw_cols = [k for k in df_features.keys() if k not in list(dfdecay.keys())+encals_cols]
demo_cols = [k for k in df_features.keys() if k in list(dfdecay.keys())]
df_allfeatures = df_class_feats.merge(pd.concat([dfcens.loc[:,'id'],dfdecay.loc[:,decay_cols]],axis=1),on='id',
                                      how='outer').merge(pd.concat([dfcens.loc[:,'id'],df_features],axis=1),
                                                         on='id',how='outer')

# get all the of the first timepoint slopes
df_allfeatures = df_allfeatures.rename(columns={'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS':'Slope_Onset_to_FirstBulbar',
                                                'ALSFRS_Slope_Onset_to_FirstALSFRS':'Slope_Onset_to_FirstALSFRS'})# renaming to keep consistency

# % FEV 
cols_fev = [c for c in df_allfeatures.columns if "_% predicted" in c and c.startswith('d')]
df_allfeatures['first_fev'] = df_allfeatures.loc[:,cols_fev].fillna(method='bfill', axis=1).iloc[:, 0]
# df_allfeatures['mean_fev'] = df_allfeatures.loc[:,'d0_% predicted':'d1080_% predicted'].mean(axis=1)# older version
df_allfeatures.loc[:,r'Slope_Onset_to_First%FEV'] = (100-df_allfeatures.loc[:,cols_fev].fillna(method='bfill', axis=1).iloc[:, 0])/(df_allfeatures.loc[:,cols_fev].notna().idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

# ALScols = [c for c in df_allfeatures.keys() if c.find('_ALSFRS_Total')!=-1 and c.startswith('d')]
# df_allfeatures.loc[:,ALScols]
# df_allfeatures.loc[:,ALScols].fillna(method='bfill', axis=1).iloc[:, 0]

# # ALSFRS_Slope_Onset_to_FirstWeight
# cols_weight = [c for c in df_allfeatures.columns if "Weight" in c and c.startswith('d')]

# mask_not_na = df_allfeatures.loc[:,cols_weight].notna()
# first_col = mask_not_na.idxmax(axis=1)
# first_row = mask_not_na.idxmax(axis=0)
# mask_second = mask_not_na.copy()
# for idx, col in first_col.items():
#     mask_second.loc[idx, col] = False
# second_col = mask_second.idxmax(axis=1)
# second_row = mask_second.idxmax(axis=0)
# df_allfeatures.loc[:,r'ALSFRS_Slope_Onset_to_FirstWeight'] = (4-df_allfeatures.loc[:,cols_weight].fillna(method='bfill', axis=1).iloc[:, 0])/(.idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

# ALSFRS_Slope_Onset_to_Firstq3
cols_q3 = [c for c in df_allfeatures.columns if "q3" in c and c.startswith('d')]
df_allfeatures.loc[:,r'Slope_Onset_to_FirstQ3'] = (4-df_allfeatures.loc[:,cols_q3].fillna(method='bfill', axis=1).iloc[:, 0])/(df_allfeatures.loc[:,cols_q3].notna().idxmax(axis=1).apply(lambda x: re.search(r"d(\d+).+" , x).group(1)).astype(float)+0.01)

# df_encals = df_features.loc[:,encals_cols]
pd.DataFrame(df_allfeatures.columns, columns=['column_name']).to_csv('data/allfeature_columns.csv', index=False)
df_allfeatures.to_csv('data/allfeatures.csv',index=False)

dfcens.to_csv('data/cens_death.csv',index=False)
dfcens.drop(columns=['Dead','Death_Date'],inplace=True)
dfcens.to_csv('data/cens.csv',index=False)

# calculate the feature keys
featurekey = pd.DataFrame(raw_cols,columns=['feature'])
featurekey['type'] = 'raw'
featurekey2 = pd.DataFrame(decay_cols,columns=['feature'])
featurekey2['type'] = 'decay'
featurekey3 = pd.DataFrame(df_class_feats.iloc[:,1:].keys(),columns=['feature'])
featurekey3['type'] = 'class'
featurekey3 = featurekey3.query('feature != "weight_0"').reset_index(drop=True)
featurekey4 = pd.DataFrame(demo_cols,columns=['feature'])
featurekey4['type'] = 'demo'
featurekey4 = featurekey4.loc[~featurekey4['feature'].isin(['ALSFRS_Slope_Onset_to_FirstALSFRS',
                                                            'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS'])].reset_index(drop=True)
featurekey5 = pd.DataFrame(encals_cols,columns=['feature'])
featurekey5['type'] = 'encals'  
featurekey6 = pd.DataFrame(['Slope_Onset_to_FirstALSFRS',
            'Slope_Onset_to_FirstBulbar',
            r'Slope_Onset_to_First%FEV',
            'Slope_Onset_to_FirstQ3'],columns=['feature'])
featurekey6['type'] = 'onset_slope'
featurekey = pd.concat([featurekey,featurekey2,featurekey3,featurekey4,featurekey5,featurekey6],axis=0,ignore_index=True)
featurekey.to_csv('data/featurekey.csv',index=False)

