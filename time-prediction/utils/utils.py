# constants.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# These are general utilities

import pickle 
import os
import pandas as pd
import copy
from utils.constants import *
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import altair as alt
from frechetdist import frdist
from lifelines.utils import concordance_index
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score,roc_curve, roc_auc_score

fulldatapath = "/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data"

def exp_decay_fcn(x, slope, maxval=48):
    return maxval * np.exp (- x / slope )

def lin_fcn (x, slope, maxval):
    return maxval - x * slope

def median_absolute_error(y_test,y_pred):
    return np.median(np.abs(y_test-y_pred))

def mean_absolute_error(y_test,y_pred):
    return np.mean(np.abs(y_test-y_pred))

def cal_predintol(y_test,y_pred,offset=90):
    return (np.abs(y_test-y_pred)<=offset).sum()/y_pred.shape[0]

def cal_metrics(y_train,y_pred,t_po=None):
    if isinstance(y_train,pd.DataFrame):
        y_train = y_train.values.reshape(-1,)
    if isinstance(y_train,list):
        y_train = np.array(y_train)
    
    return {"MSE":mean_squared_error(y_train,y_pred),
            "RMSE":np.sqrt(mean_squared_error(y_train,y_pred)),
            "R2":r2_score(y_train,y_pred),
            "MAE":mean_absolute_error(y_train,y_pred),
            "MedianAE":median_absolute_error(y_train,y_pred),
            "MedianAEPO":median_absolute_error(t_po.reshape(-1,),y_pred.reshape(-1,)) if t_po is not None else None,
            "MeanAEPO":mean_absolute_error(t_po.reshape(-1,),y_pred.reshape(-1,)) if t_po is not None else None,
            # "PredIn90":cal_predintol(y_train,y_pred,offset=90),
            # "PredIn180":cal_predintol(y_train,y_pred,offset=180),
            # "PredIn360":cal_predintol(y_train,y_pred,offset=360),
            "Cindex":concordance_index(y_train,y_pred)}

def metrics_wrapper(results,res_df,y_train,y_pred,id_train,offset = 90,name='RF',data_type='train'):
    aux = pd.DataFrame(cal_metrics(y_train,y_pred,offset),index=[0])
    aux['model'] = name
    aux['date'] = pd.Timestamp.now()
    aux['data'] = data_type
    # build an individual results dataframe
    df = pd.DataFrame([id_train.values,y_train.values.reshape(-1,),y_pred],columns=range(len(y_train)),index = ['id','y_true','y_pred']).T
    df['data'] = data_type
    df['model'] = name
    df['date'] = pd.Timestamp.now()
    df['cil'] = df['y_true'] - offset
    df['ciu'] = df['y_true'] + offset
    df['n'] = df.shape[0]
    return pd.concat([results,aux],axis=0,ignore_index=True),pd.concat([res_df,df],axis=0,ignore_index=True),

def add_overfit_margin(results):
    results['model'] = results['model'].apply(lambda x:"_".join(x.split("_")[:-1]))
    aux = results.groupby('model').apply(lambda x: x.loc[x['data']=='train','Acc'].values[0]-x.loc[x['data']=='test','Acc'].values[0]).reset_index()
    aux = aux.rename(columns={0:'overfit'})
    results = results.merge(aux,on='model',how='left')
    results['Acc_adj'] = results['Acc']-results['overfit']
    results.sort_values(['data','Acc_adj'],ascending=[True,False],inplace=True)
    # reorder columns
    col2 = results.pop('Acc_adj')
    results.insert(results.shape[1]-4, 'Acc_adj', col2)
    col2 = results.pop('overfit')
    results.insert(results.shape[1]-4, 'overfit', col2)
    return results

def plot_all(res_df,name='results',model_subset=None):
    # select subset of models
    res_df['model'] = res_df['model'].apply(lambda x:"_".join(x.split("_")[:-1]))
    res_df = res_df.query(f'model in {tuple(model_subset)}').reset_index(drop=True) if model_subset else res_df
    res_df['yerror'] = res_df['y_pred']-res_df['y_true']
    res_df['in'] = (np.abs(res_df['yerror'])<=90).map({True:'in',False:'out'})
    res_df['in'] = res_df['data']+res_df['in']

    res_df = id_to_numeric(res_df)
    res_df['id'] = res_df['cil']+(res_df['data']=='train')*100000
    res_df = res_df.query('y_pred<=1080 and y_true<=1080').reset_index(drop=True)   
    # res_df.sort_values(['cil'],inplace=True)

    idx = (res_df.loc[:,['y_pred','y_true','cil','ciu']]>=0).all(axis=1)
    res_df = res_df.loc[idx,:]
  
    base = alt.Chart(res_df).encode(y=alt.Y("id:N", axis=alt.Axis(labels=False)))
    bar = base.mark_errorbar().encode(
        x=alt.X("ciu:Q",scale=alt.Scale(domain=[0, 1080])).title("days"),
        x2=alt.X2("cil:Q"),
        color = alt.value('lightgray')
    )
    dots = base.mark_circle().encode(x=alt.X('y_pred',scale=alt.Scale(domain=[0, 1080])),
                                # color = alt.value('red'),
                                    color = alt.Color('in',scale=alt.Scale(domain=['testin','testout','trainin','trainout'],
                                                                           range=['#399918','#FF7777','#ECFFE6','#FFAAAA'])),
                                    tooltip=['y_true','y_pred','yerror','id','model','data'])

    chart = (bar + dots).properties(width=200,height=500).facet(column='model')
    chart.save(f'data/{name}.html')

def get_raw_feartures(df_features,dfcens):
    # encode features
    # categorical
    cat_features = ['site_onset','Phenotype','Database','sex']
    df_features = pd.get_dummies(df_features,columns=cat_features)

    # encode boolean
    bool_cols = ['C9orf72', 'SOD1',  'FUS', 'TARDBP']
    for col in bool_cols:
        df_features.loc[:,col] = df_features.loc[:,col].map({'POSITIVE':1,'NEGATIVE':1})

    # drop the id columns
    ids = df_features.loc[:,'id']
    df_features.drop(columns=['id','_merge'],inplace=True)

    X = df_features
    y = dfcens.loc[:,['id','tcens']]
    return X,y,ids,df_features

def calculate_first_FEV(df_master):
    cols_fev = [k for k in df_master.keys() if k.find('% predicted')!=-1]
    oo = df_master.loc[:,cols_fev]
    aa = pd.DataFrame(np.where(oo.notna())).T.drop_duplicates(subset=[0])
    return pd.DataFrame([oo.iloc[row[0],row[1]] for r,row in aa.iterrows()],columns=['first_FEV'],index=aa[0])

def infer_new_ALSFRS_slope(df_master,dffeat):
    # this is the ALSFRS slope based on diagnostic delay in months as per encals model
    cols_alsfrs = [k for k in df_master.keys() if k.endswith('ALSFRS_Date')]
    aux = df_master.loc[:,cols_alsfrs].reset_index()
    aux = aux.melt(id_vars='index').dropna(subset=['value']).sort_values('value',ascending=True).drop_duplicates(subset=['index'],keep='first')
    aux.index = aux['index']
    aux = aux.drop(columns=['variable','index']).rename(columns={'value':'ALSFRS_days'})
    dffeat = pd.concat([dffeat,aux],axis=1)
    dffeat['ALSFRS'] = 48-dffeat['ALSFRS_Slope_Onset_to_FirstALSFRS']*dffeat['ALSFRS_days']
    dffeat['ALSFRS_Slope_dxdelay'] = (48-dffeat['ALSFRS'])/dffeat['diagnostic_delay_months']
    return dffeat

def get_initial_features(df_master,df_phenotype_mapper):
    # get the time-dependent variables: 
    cols = ['id','Phenotype'] + [c for c in df_master.columns if c.startswith('d') and (c.endswith('Weight') or c.endswith('bulbar_subscore') or c.endswith('ALSFRS_Total') or c.endswith('q3') or c.endswith('% predicted'))] 
    dftd = df_master.loc[:,cols]

    # get the time-independent variables:
    cols = ['Database', 'id', 'sex', 'age_at_onset', 'site_onset','el_escorial',
        'diagnostic_delay_months', 'ALSFRS_Slope_Onset_to_FirstALSFRS',
        'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS','C9orf72','SOD1', 'FUS','TARDBP']
    dfti = df_master.loc[:,cols]
    dfti['Clinical'] = dfti['Database'].isin(['PROACT'])

    df_features = pd.merge(dftd,dfti,on='id',how='outer',indicator=True)

    # get censoring information:
    cols = ['id',  'last_follow_up',  'Outcome_Date']# I am ignoring death: 'Dead', 'Death_Date',
    # dfcens = df_master.loc[:,cols].rename(columns={'Outcome_Date':'tcens'})
    # dfcens['cens'] = dfcens['tcens'].isna()
    # dfcens['event'] = dfcens['tcens'].notna()
    # dfcens['tcens'] = dfcens['tcens'].fillna(df_master['last_follow_up'])

    dfcens = df_master.loc[:,['id','cens','tcens','Dead','Death_Date']]
    dfcens['event'] = (dfcens['cens']==0).astype(int)
    dfcens['Dead'] = dfcens['Dead'].fillna(0).astype(int)

    # dfcens = dfcens.fillna({'tcens':1080,'cens':0})
    # dfcens.loc[:,'tcens'] = dfcens.loc[:,'tcens'].clip(upper=1080)

    # remap phenotypes
    dict_phenotype_mapper = {row['Phenotype original']:row['Phenotype new'] for ri,row in df_phenotype_mapper.iterrows()}
    df_features['Phenotype'] = df_features['Phenotype'].map(dict_phenotype_mapper)
    df_features['el_escorial'] = df_features['el_escorial'].isin(['Definite','Clinically Definite']).astype(int)
    
    # get the ones with censoring information:
    df_features = df_features.merge(dfcens.loc[:,'id'],on='id',how='inner')

    ####### feature engineering - this is for ENCALS death model
    df_fev = calculate_first_FEV(df_master)

    cols_death_encals = ['age_at_onset','C9orf72','site_onset','ALSFRS_Slope_Onset_to_FirstALSFRS',  'el_escorial','diagnostic_delay_months','Dead', 'Death_Date','Database']
    df_encals = pd.concat([df_fev,df_master.loc[:,cols_death_encals]
                            ],axis=1)

    df_encals['FTD'] = 0

    # reformat further
    df_encals['site_onset'] = (df_encals['site_onset']=='Bulbar').astype(int)
    df_encals['C9orf72'] = (df_encals['C9orf72']=='POSITIVE').astype(int)
    df_encals['el_escorial'] = df_encals['el_escorial'].isin(['Definite','Clinically Definite']).astype(int)

    df_to_impute = df_encals.drop(columns=['Dead', 'Death_Date','Database'])

    # MICE imputation - Multiple Imputation by Chain Equations
    imp_mean = IterativeImputer(random_state=1,n_nearest_features=10,imputation_order='random')
    aux = pd.DataFrame(imp_mean.fit_transform(df_to_impute),columns=df_to_impute.keys())

    aux = pd.concat([aux,
                        df_encals.loc[:,['Dead', 'Death_Date','Database']]],axis=1)
    aux['FTD'] = 0

    aux['Dead'] = aux['Dead'].fillna(0)
    aux = infer_new_ALSFRS_slope(df_master,aux)
    aux.to_csv('data/encals_features.csv',index=False)

    return df_features,dfcens

def load_death_features(path_master,path_encals_pred):
    # load feature data
    df_master = pd.read_csv(path_master,encoding='latin_1',low_memory=False)
    dffeat = pd.read_csv(fulldatapath+"/encals_features.csv")

    dfpred = pd.read_csv(path_encals_pred)
    dfpred.index = dfpred['Unnamed: 0']-1
    # dfpred['OUT'] *= 365.25/12
    dfgt = pd.read_csv('data/cens_death.csv')
    dfgt['Death_Date'] /= 365.25/12
    # dfpred['OUT'] /= 365.25/12

    # dffeat['ALSFRS'] = 48-dffeat['ALSFRS_Slope_Onset_to_FirstALSFRS']*365.15/12*dffeat['diagnostic_delay_months']
    df = pd.concat([dfpred,dfgt,dffeat.loc[:,'Database']],axis=1)

    df['tcens'] /= 365.25/12
    idx = df.loc[:,['OUT','Death_Date']].notna().all(axis=1)
    df_unc = df.loc[idx,:]
    

    res_encals_death = []
    for li,l in list(df_unc.groupby('Database')):
        results = cal_metrics(l.loc[:,'Death_Date'].values,l.loc[:,'OUT'].values)
        results.update({'n':l.shape[0]})
        print(li,results)
        res_encals_death.append(results)
    dffeat.loc[df_unc.index,:].query('C9orf72 == 1')

    # aa = np.array(df_unc.index)
    # np.random.shuffle(aa)
    # aa[:10]

    # # selected: 0, 2, 3 , 8581, 8585, 6000, 6004, 6023,8525,4715
    selected = [3 , 3574, 8191, 5274, 58]
    aa = dffeat.loc[selected,:]

    ao = df_unc.loc[selected,['Death_Date','OUT']].rename(columns={'OUT':'encals_R_script'})
    ao['manual_online_tool'] = [18,15,15,34,20]
    print(ao)
    
    dffeat = pd.concat([dffeat,df_master.loc[:,['tcens','Database']]],axis=1)
    dffeat.loc[:,'Death_Date'] = dffeat.loc[:,'Death_Date'].fillna(dffeat.loc[:,'tcens'])
    dffeat = dffeat.query('Death_Date >= 0').reset_index(drop=True)

    y = np.array([(int(row['Dead']),row['Death_Date']) for ri,row in  dffeat.loc[:,['Dead','Death_Date']].iterrows()],
                dtype=[('dead', '?'), ('time', '<f8')])
    X = dffeat.loc[:,:'FTD'].values
    return X, y,dffeat,df

# transform id into numeric ids
def id_to_numeric(df):
    df['id'] = pd.factorize(df['id'])[0]
    return df

def exp_decay(x,a,b):
    return a*np.exp(-b*x)

def deconv_id(y_train,y_test):
    id_test = y_test.loc[:,'id']
    y_test.drop(columns=['id'],inplace=True)
    id_train = y_train.loc[:,'id']
    y_train.drop(columns=['id'],inplace=True)
    return y_train,y_test,id_train,id_test

def verticalise(daux):
    D = []
    for c in daux.keys()[1:]:
        aux = daux.loc[:,['id',c]].rename(columns={c:'value'})
        aux['days'] = c.split('_')[0].replace('d','').replace('p','')
        aux['extra_days'] = c.split('_')[0].find('p')!=-1
        aux['variable'] = '_'.join(c.split('_')[1:])
        aux.dropna(subset=['value'],inplace=True)
        D.append(aux.reset_index(drop=True))
        print(c,aux.keys())
    return pd.concat(D,axis=0,ignore_index=True)

def verticalise_classes(df_classes):
    df_class_feats2 = df_classes.melt(id_vars=['id']).query('value != 0').reset_index(drop=True)
    df_class_feats2['class'] = df_class_feats2['variable'].apply(lambda x:x.split('_')[-1])
    df_class_feats2['variable'] = df_class_feats2['variable'].apply(lambda x:'_'.join(x.split('_')[:-1]))
    return df_class_feats2

def convert_to_class_means(df_classes,df_features):
    df_features_v = verticalise(df_features.drop(columns=['Phenotype']+list(df_features.iloc[:,-14:].columns)))
    df_features_v['days'] = df_features_v['days'].astype(int)
    df_classes = verticalise_classes(df_classes)
    A = []
    for var in df_classes['variable'].unique():#.iloc[:,1:].columns:
        for li,l in list(df_classes.query(f'variable == "{var}"').groupby('class')):#list(df_classes.loc[:,['id',var]].groupby(var)):
            var2 = 'Weight' if var == 'weight' else var
            daux = df_features_v.query(f"id in {tuple(l['id'].values.astype(int).tolist())} and variable == '{var2}'")
            g = daux.groupby('days')['value']
            A.append(pd.concat([g.mean().rename(f'{var}_{li}'),
                        (g.mean()+1.96*g.std()/np.sqrt(g.count())).rename(f'{var}_{li}_ul'),
                        (g.mean()-1.96*g.std()/np.sqrt(g.count())).rename(f'{var}_{li}_ll')],axis=1))
    A = pd.concat(A,axis=1).reset_index().sort_values(['days']).reset_index(drop=True)
    A = np.clip(A,0,np.inf)
    return A

def get_trendlines(df_classes_means,reload = False):
    
    if reload:
        dpopt = []
        cols = [k for k in df_classes_means.keys() if (not k.endswith('_ul') and not k.endswith('_ll')) and k != 'days']
        for var in cols:
            varname = '_'.join(var.split("_")[:-1])
            dfnow = df_classes_means.loc[:,['days',var]].dropna()
            dfnow = pd.concat([pd.DataFrame([0,maxvals[varname]],index=dfnow.keys(),columns=[0]).T, dfnow]).reset_index(drop=True)
            print(var)
            popt, pcov = curve_fit(exp_decay, dfnow['days'].astype(float), dfnow.loc[:,var],p0=[1,0.01])
            dpopt.append(pd.DataFrame(["_".join(var.split("_")[:-1]),var.split("_")[-1]]+popt.tolist()+[pcov[0,0]**.5,pcov[1,1]**.5],index=['variable','class','a','b','a_stdE','b_stdE']).T)
        dpopt = pd.concat(dpopt,axis=0)
        dpopt.to_csv('data/class_exp_decay_fits.csv',index=False)
        return dpopt
    else:
        return pd.read_csv('data/class_exp_decay_fits.csv')

def get_decay_features(dfcens,df_features,ids,reload_popt=False):
    daux = pd.concat([ids,df_features.loc[:,'d0_q3':'d1080p_% predicted']],axis=1)
    daux = verticalise(daux)

    # make wide format
    daux = daux.pivot_table(index=['id','days'],columns='variable',values='value')

    daux = daux.ffill().fillna(0)

    daux = daux.reset_index()

    # fit exponential decay
    if reload_popt:
        Popt = []
        for id in daux['id'].unique():
            daux2 = daux.loc[daux['id']==id,:]
            for var in daux.columns[1:]:
                try:
                    print(var)
                    popt, pcov = curve_fit(exp_decay, daux2['days'].astype(float), daux2.loc[:,var],p0=[1,0.01])
                    Popt.append([id,var]+popt.tolist())
                except:
                    print('error',var,id)
                    Popt.append([id,var,np.nan,np.nan])

        dfpopt = pd.DataFrame(Popt,columns=['id','variable','a','b'])

        # pivot data
        dfpopt = dfpopt.pivot_table(index='id',columns='variable',values=['a','b'])
        # merge indexes
        dfpopt.columns = ['_'.join(c) for c in dfpopt.columns]
        dfpopt = dfpopt.reset_index()

        dfpopt.to_csv('data/decayfeatures.csv',index=False)
    else:
        dfpopt = pd.read_csv('data/decayfeatures.csv')

    # merge the data
    daux = pd.concat([ids,df_features.loc[:,'age_at_onset':]],axis=1)

    # RF regressor
    X = daux.merge(dfpopt,on='id',how='outer')
    X.drop(columns=['id'],inplace=True)
    y = dfcens.loc[:,['id','tcens']]
    return X,y

def run_train_eval_pipe(X,y,dfstrat,results,res_df,model,name='RF',feature_names='ALSFRS',early_stopping=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=dfstrat)

    y_train,y_test,id_train,id_test = deconv_id(y_train,y_test)

    if feature_names == 'ALSFRS':
        X_train = X_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)

    pipe = make_pipeline(StandardScaler(),model)
    
    if early_stopping:
        eval_set = [(pipe[0].fit_transform(X_test), y_test)]
        # model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
        pipe.fit(X_train,y_train, xgbregressor__eval_set=eval_set, xgbregressor__verbose=True)
    else:
        pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_train)
    results,res_df = metrics_wrapper(results,res_df,y_train,y_pred,id_train,offset = 90,name=f'{feature_names}_{name}_train',data_type='train')
    print(f"{feature_names} {name} train",cal_metrics(y_train,y_pred,90))
    y_pred = pipe.predict(X_test)
    results,res_df = metrics_wrapper(results,res_df,y_test,y_pred,id_test,offset = 90,name=f'{feature_names}_{name}_test',data_type='test')
    print(f"{feature_names} {name} test",cal_metrics(y_test,y_pred,90))
    return results,res_df

def compute_frechet_distance(dfpoint,var="ALSFRS"):    
    P = dfpoint.loc[:,["days",var]].astype(int).values.tolist()
    Q = [dfpoint.loc[:,["days",f"{var}_inferred_{ni}"]].astype(int).values.tolist() for ni in range(1,classnums[var]+1)]
    out = [frdist(P,q) for q in Q]
    if len(out)<4:
        out += [np.nan]*(4-len(out))
    return out

def compute_frobenius_norm(dfpoint,var="ALSFRS"):    
    P = dfpoint.loc[:,["days",var]].astype(int).values.tolist()
    Q = [dfpoint.loc[:,["days",f"{var}_inferred_{ni}"]].astype(int).values.tolist() for ni in range(1,classnums[var]+1)]
    return [np.sum((np.array(q)-np.array(P))**2)**.5 for q in Q]

def compute_all_distances(df_features_v,dpopt):
    distances = []
    for var in ['q3', 'bulbar_subscore', 'ALSFRS_Total', '% predicted']:
        
        dfnow = df_features_v.query('variable == @var').copy().dropna(subset=['value','days']).rename(columns={'value':var})
        popt = dpopt.query(f'variable == "{var}"')
        for ri, poptnow in popt.iterrows():
            dfnow[f'{var}_inferred_'+poptnow['class']] = dfnow[var].apply(lambda x:exp_decay(x,poptnow['a'],poptnow['b']))
        
        for li,L in list(dfnow.groupby('id')):
            distances.append([var,li]+compute_frechet_distance(L,var=var)+compute_frobenius_norm(L,var=var))

    return pd.DataFrame(distances,columns=['var','id']+["class_"+str(k) for k in range(1,5)]+["frob_"+str(k) for k in range(1,5)]) 

def map_classes(distances):
    out = []
    for li,l in distances.groupby('var'):
        classes = ["class_"+str(c) for c in range(1,classnums[li]+1)]
        # get the classes that are the same and apply a random tie breaker
        mins = l.loc[:,classes].min(axis=1)
        id = l.loc[:,classes].values == np.repeat(mins.values.reshape(-1,1), classnums[li],axis=1)
        id2 = pd.DataFrame(np.where(id)).T.sample(frac=1).drop_duplicates(subset=0,keep='first').sort_values(0)[1]
        l['class'] = id2.values+1
        out.append(l.loc[:,['var','id','class']])
    return pd.concat(out,axis=0)

def get_class_features_frechet(df_classes,df_features,suffix = "",reload_class_features=False):
    if reload_class_features:
        # get the class means
        df_classes_means = convert_to_class_means(df_classes,df_features)

        # get trendlines
        dpopt = get_trendlines(df_classes_means,reload=True)

        # reformat features
        df_features_v = verticalise(df_features.drop(columns=['Phenotype']+list(df_features.iloc[:,-14:].columns)))

        # get the fretchet distances
        distances = compute_all_distances(df_features_v,dpopt)

        # map classes
        df_class_pred = map_classes(distances)
        df_class_feats = pd.get_dummies(df_class_pred.pivot(index='id',columns='var',values='class').fillna(0).astype(int),columns=['ALSFRS_Total',  'bulbar_subscore',  'q3',  '% predicted']).astype(int).reset_index()
        cols_to_drop = [c for c in df_class_feats.keys() if c.endswith('0')]
        df_class_feats.drop(columns=cols_to_drop,inplace=True)
        df_class_feats.to_csv(f'data/class_features{suffix}.csv',index=False)
        return df_class_feats
    else:
        return pd.read_csv(f'data/class_features{suffix}.csv')

def load_classes():
    df_classes = []
    var_dict_lower = {k.lower():v for k,v in var_dict.items()}
    for ki,k in enumerate(os.listdir(fulldatapath+'/classes')):
        aux = pd.read_csv(os.path.join(fulldatapath+'/classes', k))
        variable_name = var_dict_lower[k.replace('dt_','').replace('.csv','').lower()]
        aux['variable'] = variable_name
        aux = aux.rename(columns={'numID':'id','cl'+str(classnums[variable_name]):'class'})
        cols = ['id','variable','class']
        df_classes.append(aux.loc[:,cols])
    df_classes = pd.concat(df_classes)
    # add the weight classes
    dfw = pd.read_csv(fulldatapath+'/weight_delay_dataset_cens.csv').query('threshold == 0.05')
    dfw = dfw.loc[:,['numid','threshold_reached']].rename(columns={'numid':'id','threshold_reached':'class'})
    dfw['variable'] = 'weight'
    return pd.concat([df_classes,dfw],axis=0,ignore_index=True)
    
def get_class_features_gt():
    df_classes00 = load_classes()
    df_classes0 = reformat_classes(df_classes00)
    df_classes = pd.get_dummies(df_classes0).reset_index().astype(int)
    cols_to_drop = [c for c in df_classes.keys() if c.endswith('0') and not c.startswith('weight')]
    df_classes.drop(columns=cols_to_drop,inplace=True)
    return df_classes

def reformat_classes(df_classes00):
    return pd.pivot_table(df_classes00,index='id',columns=['variable'],values='class').fillna(0).astype(int).astype(str)

def fetch_slope_points(var,myclass,timedomain):
        dfauxx = pd.DataFrame([timedomain],index=['time']).T
        slope = funpar[var][myclass][0]
        maxval = maxvals[var]
        dfauxx["inferred_mean"] = dfauxx["time"].apply(lambda x:globals()[fcn_var[var]](x, slope, maxval=maxval))
        dfauxx["inferred_overall_mean"] = dfauxx["inferred_mean"].mean()
        return dfauxx

def recover_std(dfnow,myclass,time):
    aa = dfnow.query(f'`class` == {myclass} and time == {time}')
    # if the value does not exist, take the closest one
    if aa.empty: 
        aa = dfnow.query(f'`class` == {myclass}')
        aa['t_diff'] = np.abs(aa['time']-time)
        aa = aa.sort_values(by='t_diff').iloc[0,:]
    return aa['mean']-aa['lowCI']

def add_ordinal(df_classes0,ignore_weight=False):
    df_class_feats2 = verticalise_classes(df_classes0).copy()
    # print("shape1",df_class_feats2.shape)
    df_classes1 = df_class_feats2.pivot(index='id',columns=['variable'],values='class')
    df_classes1 = df_classes1.rename(columns={c:c+"_ord_cl_eq" for c in df_classes1.keys() if c != 'id'}).reset_index()
    df_classes = df_classes0.merge(df_classes1,on='id',how='outer')
    # print("shape2",df_class_feats2.shape)
    # add ordinal classes - maximal
    class_dict = {'q3': 'q3', 'Bulb': 'bulbar_subscore', 'Resp': '% predicted', 'TALS': 'ALSFRS_Total', 'Weight': 'weight'}
    # df_classes = df_classes.merge(df_classes0,on='id',how='outer')
    dff = pd.read_csv(fulldatapath+'/class_traj_summary.csv')
    AA = {}
    for var,l in list(dff.groupby(['variable'])):
        if ignore_weight and var[0]=='Weight':
            continue
        dfnow = l.query('time == 540')
        A = {str(row['class']):row['mean']/l['mean'].max() for ri,row in dfnow.iterrows()}
        dfclassnow = copy.deepcopy(df_class_feats2.query(f'variable == "{class_dict[var[0]]}"'))
        dfclassnow.loc[:,class_dict[var[0]]+'_ord_cl_neq'] = dfclassnow.loc[:,'class'].map(A)
        df_classes = df_classes.merge(dfclassnow.loc[:,['id',class_dict[var[0]]+'_ord_cl_neq']],on='id',how='outer')
        # print("shape3",l[0][0],df_class_feats2.shape)

        # the longitudinal slope
        timedomain= tuple(np.arange(90,1170,90).tolist())
        dfnow = l.query(f'time in {timedomain}')
        A = {str(lo.iloc[0,5]):[(maxvals[class_dict[row['variable']]]-row['mean'])/row['time']*365.25/12 for ri,row in lo.iterrows()] for li, lo in list(dfnow.groupby('class'))}
        dfclassnow = copy.deepcopy(df_class_feats2.query(f'variable == "{class_dict[var[0]]}"'))
        dfclassnow.loc[:,class_dict[var[0]]+'_ord_cl_slope'] = dfclassnow.loc[:,'class'].map(A)
        columns = [class_dict[var[0]]+'_ord_cl_slope_' + str(t) for t in timedomain]
        dfclassnow.dropna(subset=[class_dict[var[0]]+'_ord_cl_slope'],inplace=True)
        dfclassnow[columns] = pd.DataFrame(dfclassnow.loc[:,class_dict[var[0]]+'_ord_cl_slope'].tolist(), 
                                                                                                       index= dfclassnow.index)
        df_classes = df_classes.merge(dfclassnow.loc[:,['id']+columns],on='id',how='outer')
        
        # neqm - longitudinal value
        timedomain= tuple(np.arange(0,1170,90).tolist())
        dfnow = l.query(f'time in {timedomain}')
        # this was a test to jitter the results but it is not resolving anything
        ##### IMPORTANT NOTE: A was indexes to be just one number [0] not the array - this needs to be verified
        A = {str(myclass):[float(np.random.normal(row['inferred_mean'], recover_std(dfnow,myclass,row["time"]))[0])/dfnow['mean'].mean() for ri,row in fetch_slope_points(class_dict[var[0]],myclass,timedomain).iterrows()] for myclass in l['class'].unique()}
        # A = {str(myclass):[row['inferred_mean']/dfnow['mean'].mean() for ri,row in fetch_slope_points(class_dict[var[0]],myclass,timedomain).iterrows()] for myclass in l['class'].unique()}
        dfclassnow = copy.deepcopy(df_class_feats2.query(f'variable == "{class_dict[var[0]]}"'))
        dfclassnow.loc[:,class_dict[var[0]]+'_ord_cl_neqm'] = dfclassnow.loc[:,'class'].map(A)
        columns = [class_dict[var[0]]+'_ord_cl_neqm_' + str(t) for t in timedomain]
        dfclassnow.dropna(subset=[class_dict[var[0]]+'_ord_cl_neqm'],inplace=True)
        dfclassnow[columns] = pd.DataFrame(dfclassnow.loc[:,class_dict[var[0]]+'_ord_cl_neqm'].tolist(), 
                                                                                                       index= dfclassnow.index)
        df_classes = df_classes.merge(dfclassnow.loc[:,['id']+columns],on='id',how='outer')
        AA.update({var:A})

    with open('data/neqm_values.pkl', 'wb') as f:
        pickle.dump(AA, f)
        
    # A = {1:l[1]['mean'].max()-dfnow['mean'].max()}
    # A.update({M:(dfnow.query(f'`class` == {M}')['mean'].values-dfnow.query(f'`class` == {M-1}')['mean'].values)[0] for M in range(2,dfnow['class'].max()+1)})
    return df_classes

def plot_model_results(results,model_subset):
    results2 = results.query(f'model in {tuple(model_subset)}').reset_index(drop=True) 
    results2 = pd.melt(results2,id_vars=['model','data'],
                    value_vars=['MSE', 'RMSE', 'R2', 'MAE', 'Acc', 'Cindex', 'overfit', 'Acc_adj'])
    results2['value'] = results2['value'].round(2)
    base = alt.Chart(results2).encode(y='model',
                                        x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
                                    color=alt.Color('data',scale=alt.Scale(domain=['test','train'],range=['#399918','#9CDBA6'])),
                                    text = 'value')

    (base.mark_bar()+base.mark_text(align='left', dx=2)).properties(width=100,height=100).facet(column='variable:N',row='data').resolve_scale(x='independent').save('data/results_metrics.html')

def features_km(df_master):
    cols_km = ['Database',     'id' ,  'tcens' , 'cens' ,  'Phenotype', 'site_onset', 'sex' , 'age_at_onset','Baseline_Weight','Dead','Death_Date', 'last_follow_up',  'Outcome_Date','Onset_Date']
    aux = df_master.loc[:,cols_km].copy()
    aux['Weight_groups'] = pd.cut(aux['Baseline_Weight'],bins=[0,50,80,100,np.inf],labels=['<50','50-80','80-100','100+']).astype(str)
    aux['age_groups'] = pd.cut(aux['age_at_onset'],bins=[0,50,70,np.inf],labels=['<50','50-70','70+']).astype(str)
    aux['tcens_corr'] = (aux['Outcome_Date'] - aux['Onset_Date']).fillna(aux['last_follow_up'] - aux['Onset_Date'])
    aux['cens_corr'] = aux['Outcome_Date'].isna().astype(int)
    aux.to_csv('data/km.csv',index=False)

def fit_logistic_reg_death_gastro(df_combo):
    dfnew = df_combo.query('lbl in ("Gastrostomy","Death")').loc[:,['lbl','gastrostomy_pred','death_pred']]
    
    dfnew = dfnew.dropna()
    X, y = dfnew.loc[:,['gastrostomy_pred','death_pred']].values, (dfnew['lbl']=='Gastrostomy').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = LogisticRegression(random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=1) 
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation CV Accuracy: {np.std(cv_scores):.4f}")

    y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate AUC (Area Under the ROC Curve)
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:",roc_auc)
    
    return clf,fpr,tpr

def fit_rbf_svm_death_gastro(df_combo):
    from sklearn import svm
    dfnew = df_combo.query('lbl in ("Gastrostomy","Death")').loc[:,['lbl','gastrostomy_pred','death_pred']]
    
    dfnew = dfnew.dropna()
    X, y = dfnew.loc[:,['gastrostomy_pred','death_pred']].values, (dfnew['lbl']=='Gastrostomy').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    C = 1  # SVM regularization parameter
    clf = svm.SVC(kernel = 'rbf',  gamma='auto', degree=2, C=C , probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=1) 
    cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    print(f"Cross-Validation Accuracy (5-fold): {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation CV Accuracy: {np.std(cv_scores):.4f}")

    y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities of the positive class

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # Calculate AUC (Area Under the ROC Curve)
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:",roc_auc)
    
    return clf,fpr,tpr

def compute_roc_auc_threshold(df_combo):
    df_combo = df_combo.query('lbl in ("Gastrostomy","Death")').reset_index(drop=True)
    y_test = (df_combo['lbl']=='Gastrostomy').astype(int)
    y_proba = (df_combo['gastrostomy_pred']-df_combo['death_pred'] <=0).astype(float)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print('threshold ROC AUC:',roc_auc)
    return fpr, tpr, roc_auc

def compute_roc_auc_weight(df_combo):
    df_combo = df_combo.query('lbl in ("Gastrostomy","Death")').dropna(subset='weight').reset_index(drop=True)
    y_test = (df_combo['lbl']=='Gastrostomy').astype(int)
    y_proba = (df_combo['weight']>1).astype(float)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print('weight ROC AUC:',roc_auc)
    return fpr, tpr, roc_auc


