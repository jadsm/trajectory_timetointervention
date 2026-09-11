# simulation.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file simulates different scenarios
import pickle
from utils.utils_censor import select_model_and_features,load_data,calc_all_metrics,split_data_add_mice
from utils.utils import load_death_features,cal_metrics,fit_logistic_reg_death_gastro,compute_roc_auc_threshold,fit_rbf_svm_death_gastro,compute_roc_auc_weight
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import time
import altair as alt
from utils.constants import *
import copy
from utils.utils_censor import build_competing_surv_array,get_po_times
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from itertools import combinations

logging.basicConfig(filename=f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pred.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

simulations = [4]
recompute_flag = True

def gastro_ever(df_combo):
    df_combo['pred_gastro_ever'] = df_combo['death_pred']>df_combo['gastrostomy_pred']
    df_combo['gt_gastro_ever'] = df_combo['gastro_bool']
    df_combo['acc_gastro_ever'] = df_combo['gt_gastro_ever'] == df_combo['pred_gastro_ever']
    return df_combo['acc_gastro_ever'].sum()/df_combo.shape[0]


if 0 in simulations:
# Simulation 0: weight calculation from weight_delay_dataset2 - imperial-als/fit_relu_weights_sandbox.py
    bootstraps = 100
    df = pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data/weight_delay_dataset2.csv')
    # df2 =  pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv')
    # df2 = df2.loc[:,['Database','id','tcens','cens']].rename(columns={'id':'numid'})
    df = df.query('threshold == 0.1')

    # df = df2.merge(df,on='numid',how='outer')
    
    df['delay'] = df['delay'].fillna(1080.)
    df['threshold_reached'] = df['threshold_reached'].fillna(0.)
    df['threshold'] = 0.1
    # calculate pseudo-observations
    df.loc[df['threshold_reached']==0,'delay'] = TAU#1825.
    
    df_unc = df.query('cens == 0')

    # calculate metrics of the two models
    def one_vs_rest(df,col='Database'):
        """Yield (name, group, rest) for each level of `col`: 1 vs all the others."""
        for name, group in df.groupby(col):
            yield name, group, df.loc[df[col] != name]

    def calc_metrics_wrapper(L,li,Ltrain=None):
        L = L.reset_index(drop=True)
        n = L.shape[0]
        D = []
        for _ in range(bootstraps):
            idx = np.arange(n)
            np.random.shuffle(idx)
            idx = idx[:int(n*.5)]
            ynow = build_competing_surv_array(L.loc[idx,'tcens'],L.loc[idx,'event_code'])
            t_po = get_po_times(ynow)        
            metrics_test = cal_metrics(L.loc[idx,'tcens'].to_numpy(),L.loc[idx,'delay'].to_numpy(),t_po)

            survival_train = Surv.from_arrays(time=Ltrain['tcens'], event=Ltrain['event_code']==1)
            survival_test = Surv.from_arrays(time=L.loc[idx,'tcens'].to_numpy(), event=L.loc[idx,'event_code'].to_numpy()==1)
            metrics_test.update({'Cindex_ipcw':concordance_index_ipcw(survival_train=survival_train, 
                                                                    survival_test=survival_test,
                                                estimate=L.loc[idx,'delay'], tau=TAU)[0]}) 
            dfnow = pd.DataFrame(metrics_test,index=[0])
            D.append(dfnow)
        dfnow = pd.concat(D,axis=0,ignore_index=True)
        dfnow = dfnow.describe().reset_index()
        dfnow = pd.DataFrame([dfnow.query('index=="mean"').iloc[:,1:].astype(float).values[0],
                        (dfnow.query('index=="mean"').iloc[:,1:].astype(float).values - dfnow.query('index=="std"').iloc[:,1:].astype(float).values)[0],
                      (dfnow.query('index=="mean"').iloc[:,1:].astype(float).values + dfnow.query('index=="std"').iloc[:,1:].astype(float).values)[0]],
                      index=['mean','lb','ub'],
                      columns=dfnow.iloc[:,1:].columns)
        dfnow = dfnow.T
        dfnow['test_rotation'] = li
        return dfnow
    dfout = pd.concat([calc_metrics_wrapper(L,li,Ltrain) for li,L,Ltrain in one_vs_rest(df,'Database')],axis=0)
    dfout.to_csv('data/results_weight_naive_model3.csv')

if 7 in simulations:
    from utils.utils_censor import p_value_from_ci,methods,datasets
    from itertools import combinations, product
    best_results = pd.read_csv('data/summary_database_rotation_best_last0_5Final_newclass94.csv')

# calculate pvalues
    pairs = list(combinations(list(product(methods, datasets)), 2))
    pvals = []
    for test_fold in best_results.test_fold.unique():
        for pair in pairs:
            for var in ['Cindex','MedianAE']:#,'MeanAEPO','MedianAEPO'
                a = best_results.query(f'test_fold=="{test_fold}" & method=="{pair[0][0]}" & dataset=="{pair[0][1]}"')
                b = best_results.query(f'test_fold=="{test_fold}" & method=="{pair[1][0]}" & dataset=="{pair[1][1]}"')
                pnow = p_value_from_ci(a[f'{var}_mean'].values[0],[a[f'{var}_lb'].values[0],a[f'{var}_ub'].values[0]],
                                                                b[f'{var}_mean'].values[0],[b[f'{var}_lb'].values[0],b[f'{var}_ub'].values[0]])
                pvals.append(list(pair)+[test_fold,var,pnow[0]])  
    
    pvals = pd.DataFrame(pvals,columns=['pair1','pair2','test_fold','variable','p_value'])
    pvals.to_csv('data/p_values_94.csv',index=False)

if 8 in simulations: 

    TAU = 1080
    paths = [f"data/t_vs_tpo_{TAU}.csv",f"data/t_vs_tmargin_{TAU}.csv"]
    def read_plot(path,varname = 't_po',crop_t = 2920):
        yname = "t" +varname.split('_')[1] + " (years)"
        df = pd.read_csv(path)
        df.sort_values(by='t', inplace=True)
        df[varname] = df[varname]/365.25
        df['t'] = df['t']/365.25
        df['n_gastro'] = df['event'].cumsum()
        df.reset_index(drop=True,inplace=True)
        df.reset_index(drop=False,inplace=True)
        df.rename(columns={'index': 'n_atrisk'}, inplace=True)
        df['n_atrisk'] = df['n_atrisk'].max() -df['n_atrisk']
        df['pct_atrisk'] =  df['n_atrisk']/df['n_atrisk'].max()*100
        df = df.query(f't <= {crop_t/365.25}')
        base = alt.Chart(df).encode(x=alt.X('t', axis=alt.Axis(tickCount=5),title='tcens (years)',scale=alt.Scale(domain=[df['t'].min(), df['t'].max()], nice=False)))
        bar = base.mark_line(color='gray', opacity=0.8).encode(y=alt.Y('pct_atrisk', axis=alt.Axis(orient='right', tickCount=3),title='Percentage at risk (%)'),tooltip=alt.Tooltip(['n_atrisk','t','pct_atrisk']))
        # event False -> solid, event True -> thinner dashed; Okabe-Ito colour-blind-safe blue/vermillion
        event_domain = [False,True]
        line = base.mark_line().encode(y=alt.Y(varname, axis=alt.Axis(tickCount=3, title=yname)),
                                    color=alt.Color('event:N', scale=alt.Scale(domain=event_domain, range=['#0173B2', '#DE8F05']), legend=None),
                                    strokeDash=alt.StrokeDash('event:N', scale=alt.Scale(domain=event_domain, range=[[1, 0], [5, 3]]), legend=None),
                                    strokeWidth=alt.StrokeWidth('event:N', scale=alt.Scale(domain=event_domain, range=[4, 3]), legend=None))
        # diagonal = base.mark_line(color='black', strokeDash=[5, 5], strokeWidth=1).encode(y=alt.Y('t', axis=alt.Axis(title=yname)))
        tau_line = alt.Chart(pd.DataFrame({'t': [TAU/365.25]})).mark_rule(color='black', strokeDash=[1, 1]).encode(x='t')
        chart = alt.layer(bar, alt.layer(line, tau_line)).resolve_scale(y='independent').interactive()
        chart = chart.configure_axis(labelFontSize=16, titleFontSize=16).configure_title(fontSize=20)
        chart.save(f'figures/t_vs_{varname}{TAU}.html')
        return df

    df = read_plot(paths[0],varname='t_po')
    df1 = read_plot(paths[1],varname='t_margin',crop_t = 2920)

    # 5.5% still at risk
    # TAU = 1825
    df1.drop(columns=['t'], inplace=True)
    df1.columns = [c+"_margin" if 'margin' not in c else c for c in df1.columns ]

    df2 = pd.concat([df,df1],axis=1)
    alt.Chart(df2).mark_line().encode(x='t_po',y='t_margin',color='event').save(f'figures/t_po_t_margin{TAU}.html')

if 5 in simulations:
    # descriptive analysis
    df2 =  pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv')
    idx = ~df2.loc[:,['Death_Date', 'Outcome_Date']].isna().all(axis=1)

    df2.loc[idx, 'last_follow_up'] = np.nan

    idx = df2.loc[:,['Death_Date', 'Outcome_Date']].notna().all(axis=1)

    df2.loc[idx, 'Death_Date'] = np.nan
    idx = ~(df2.loc[:,['Death_Date', 'last_follow_up',  'Outcome_Date']]<0).any(axis=1)
    df2 = df2.loc[idx,:]

    df2.loc[:,['Death_Date', 'last_follow_up',  'Outcome_Date']].describe()

    static_num_keys = ['age_at_onset', 'diagnostic_delay_months']
    df2['diagnostic_delay_months'] = pd.cut(df2['diagnostic_delay_months'],bins=[0,5,10,15,np.inf],labels=['<5','5-10','10-15','15-20'])
    df2['age_at_onset'] = pd.cut(df2['age_at_onset'],bins=[0,20,50,60,70,np.inf],labels=['<20','20-50','50-60','60-70','>70'])
    static_cat_keys = ['sex','site_onset', 'el_escorial', 'Phenotype']
    keys = ['Database','Death_Date', 'last_follow_up',  'Outcome_Date']
    df2['label'] = np.where(df2.loc[:,keys[1:]].notna())[1]
    df2['label'] = df2['label'].map({0:'death',1:'loss of follow up',2:'gastrostomy'})
    groupvar = ['Database','label']
    # df2.groupby(groupvar)[static_num_keys].mean()

    df2['Phenotype'] = df2['Phenotype'].apply(lambda x:str(x).split('_')[0])

    el_escorial_map = {'Probable Laboratory Supported':'Probable',
    'Definite':'Definite', 
    'Probable':'Probable',
        'Possible':'Possible', 
        'Clinically Definite':'Definite', 
        'Clinically Probable':'Probable',
        'Clinically lab supported':'Definite', 
        'Clinically Suspected':'Possible',
        'Clinically Possible':'Possible', 
        'ALS UMN predominant and monomelinic':'Definite'}

    df2['el_escorial'] = df2['el_escorial'].map(el_escorial_map)
    site_onset_map = {'Spinal':'Spinal', 'Bulbar':'Bulbar', 'Generalised':"Other", 'Respiratory':"Other", 'Other':"Other"}
    df2['site_onset'] = df2['site_onset'].map(site_onset_map)
    A = []
    for cat in static_cat_keys+static_num_keys:
        aux = df2.groupby(groupvar)[cat].value_counts().reset_index()
        aux['category'] = cat
        aux = aux.rename(columns={cat:'variable'})
        aux = aux.merge(aux.groupby(groupvar)['count'].sum().reset_index(),
                on=groupvar,
                suffixes=['','total'])
        aux['pct'] = aux['count']/aux['counttotal']
        A.append(aux)

    A = pd.concat(A,axis=0)

    for ci,cat in enumerate(A['category'].unique()):
        chartnow = alt.Chart(A).mark_bar().encode(x = alt.X('pct',stack=True),
                                    y=alt.Y('Database').title(None),
                                    color=alt.Color('variable').scale(scheme='category20').legend(orient='bottom',title=cat),
                                    column=alt.Column('label').title(None)).transform_filter(alt.FieldEqualPredicate(field='category',
                                                                                            equal=cat))
        if ci==0:
            chart = chartnow
        else:
            chart &= chartnow
                                                                                            
    chart.resolve_scale(color='independent').save('figures/description.html')
    # compute statistics 
    # for each database and variable - measure if they were different accorss labels
    

# column label, row category by category, x count, y  Database
if 1 in simulations:
    # Simulation 1: comparison between death vs gastrostomy model for new metric definition
    # paths = ['models/death_XGBoostReg_ArQ.pkl','models/gastro_XGBoostMAEPOReg_classes+decay_PROACT.pkl']
    # paths = ['models/gastro_XGBoostMAEPOReg_classes_ArQ.pkl']
    paths = ['models/gastro_XGBoostMAEPOReg_classes_neqm_ArQ.pkl']
    # with open(paths[0], 'rb') as fid:
    #     clf_death = pickle.load(fid)

    with open(paths[0], 'rb') as fid:
        clf_gastro = pickle.load(fid)
        
    # load data - gastro
    fixed_args = load_data()
    _,Xg,yg,strat_var,_ = select_model_and_features('classes_neqm','XGBoostMAEPOReg',fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
    _,Xg2,yg2,strat_var2,feature_names = select_model_and_features('classes','XGBoostMAEPOReg',fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
    # load data - death 
    Xd, yd,dffeat,df = load_death_features('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv',
                                           'data/encals_overall_survival_pred_final.csv')
    
    yd = yd[dffeat['tcens']>0]
    # compute gastro predictions 
    dmat = clf_gastro.dmat_builder(Xg,yg)
    y_pred_gastro = clf_gastro.final_model.predict(dmat)

    # load death predictions
    # dmat = clf_death.dmat_builder(Xd[dffeat['tcens']>0],yd)
    # y_pred_death = clf_death.final_model.predict(dmat)
    y_pred_death = df.loc[df['tcens']>0,'OUT'].values

    ### for own death model and best gastro
    yd0 = np.array([d[0] for d in yd])
    yd1 = np.array([d[1] for d in yd])
    yg0 = np.array([d[0] for d in yg])
    yg1 = np.array([d[1] for d in yg])
    df_combo = pd.DataFrame([y_pred_gastro,y_pred_death,yg0,yg1,yd0,yd1,Xg2[:,np.where(np.array(feature_names)=='weight_1')[0][0]]],index=['gastrostomy_pred','death_pred','gastro_bool','gastrostomy_gt','death_bool','death_gt','weight']).T    
    acc_xg = gastro_ever(df_combo)
    print("gastro ever accuracy (XGboost):",np.round(acc_xg,3)*100,'%')

    ## for encals death model and best gastro
    df = df.loc[df['tcens']>0,['OUT','Death_Date']]
    df_combo = pd.DataFrame([y_pred_gastro,df['OUT'].values*365.25/12,yg0,yg1,yd0,yd1,Xg2[:,np.where(np.array(feature_names)=='weight_1')[0][0]]],index=['gastrostomy_pred','death_pred','gastro_bool','gastrostomy_gt','death_bool','death_gt','weight']).T    
    # df_combo = df_combo.loc[~df_combo.loc[:,['death_bool','gastro_bool']].all(axis=1),:]
    acc_encals = gastro_ever(df_combo)
    print("gastro ever accuracy (ENCALS):",np.round(acc_encals,3)*100,'%')

    df_combo['lbl'] = df_combo['death_bool'].apply(lambda x:'Death' if x else 'Loss follow up')
    df_combo.loc[df_combo.loc[:,'gastro_bool'].values==True,'lbl'] = 'Gastrostomy'
    df_combo['gastrostomy_pred'] = df_combo['gastrostomy_pred'].astype(float).clip(lower=0, upper=13000)
    df_combo['death_pred'] = df_combo['death_pred'].astype(float).clip(lower=0, upper=13000)
    
    df_combo['height'] = 1300
    base = alt.Chart(df_combo)
    chart1 = base.mark_circle().encode(x=alt.X('gastrostomy_pred',axis=alt.Axis(style='log')),
                                                y=alt.Y('gastrostomy_gt:Q',axis=alt.Axis(style='log')),
                                                color='lbl:N')
    chart2 = base.mark_circle().encode(x=alt.X('death_pred',axis=alt.Axis(style='log')),
                                                y=alt.Y('death_gt:Q',axis=alt.Axis(style='log')),
                                                color='lbl:N'
                                                )
    line_ref = base.mark_rule(color='black').encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.value('height'),
        y2=alt.value(0)
    )
    ((chart1+line_ref).facet(row='lbl:N') | (chart2+line_ref).facet(row='lbl:N')).save('figures/death_gastro_scatter6.html')

    # likelihood inference
    c = list((df_combo['gastrostomy_pred']-df_combo['death_pred']).quantile([.33,.66]))
    df_combo['gastrostomy_proba'] = pd.cut(df_combo['gastrostomy_pred']-df_combo['death_pred'],[-np.inf]+c+[np.inf], 
                                            labels=["Likely", "Possible", "Unlikely"])
    print(f'the tertile points are:{c}')

    df_combo['gastrostomy_proba2'] = pd.cut(df_combo['gastrostomy_pred']-df_combo['death_pred'],[-np.inf,0,np.inf], 
                                            labels=["Possible", "Unlikely"])
    print(f'the two class points is: 0 days')

    clf_dg,fpr_dg,tpr_dg = fit_logistic_reg_death_gastro(df_combo)
    clf_svm,fpr_svm,tpr_svm = fit_rbf_svm_death_gastro(df_combo)

    with open(f'models/death_logistic.pkl', 'wb') as fid:
        pickle.dump(clf_dg,fid)

    dfnew = df_combo.loc[:,['lbl','gastrostomy_pred','death_pred']].dropna()
    dfnew['gastrostomy_proba3'] = clf_dg.predict(dfnew.loc[:,['gastrostomy_pred','death_pred']].values)
    dfnew['gastrostomy_proba3'] = dfnew['gastrostomy_proba3'].map({0:'Unlikely',1:'Possible'})
    df_combo = pd.concat([df_combo,dfnew['gastrostomy_proba3']],axis=1)    

    df_combo.to_csv('data/death_preds_gastro_probs6.csv',index=False)

    g = df_combo.groupby('gastrostomy_proba')
    print('probabilities of gastro by proba dist:',g['lbl'].value_counts()/g['lbl'].count())
    print('probabilities of gastro by label dist:',g['lbl'].value_counts()/df_combo['lbl'].value_counts())
    aux = (g['lbl'].value_counts()/df_combo['lbl'].value_counts()).reset_index()

    # aux.to_csv('data/gastro_ever_dist6.csv',index=False)
    
    g = df_combo.groupby('gastrostomy_proba2')
    print('probabilities of gastro by proba dist:',g['lbl'].value_counts()/g['lbl'].count())
    print('probabilities of gastro by label dist:',g['lbl'].value_counts()/df_combo['lbl'].value_counts())
    aux2 = (g['lbl'].value_counts()/df_combo['lbl'].value_counts()).reset_index()

    # aux.to_csv('data/gastro_ever_dist6.csv',index=False)

    g = df_combo.groupby('gastrostomy_proba3')
    print('probabilities of gastro by proba dist:',g['lbl'].value_counts()/g['lbl'].count())
    print('probabilities of gastro by label dist:',g['lbl'].value_counts()/df_combo['lbl'].value_counts())
    aux = (g['lbl'].value_counts()/df_combo['lbl'].value_counts()).reset_index()

    # aux.to_csv('data/gastro_ever_dist6.csv',index=False)

    # proba_palette = {'Likely':'#5AA3D1','Unlikely':'#4076B3','Possibly':'#9BC9E2'}
    proba_palette = {'Possible':'#5AA3D1','Unlikely':'#4076B3','Likely':'#9BC9E2'}

    # scatter plot - just one  and
    df_combo0 = df_combo.query('gastrostomy_gt<=1080 and death_gt<=1080').reset_index(drop=True)
    base = alt.Chart(df_combo0)
    chart3 = base.mark_circle().encode(x=alt.X('gastrostomy_pred',axis=alt.Axis(style='log')).title('predicted gastrostomy (days)'),
                                                y=alt.Y('death_pred:Q',axis=alt.Axis(style='log')).title('predicted overall survival (days)'),
                                                color=alt.Color('gastrostomy_proba3:N').title('Gastrostomy probability').scale(domain=list(proba_palette.keys()),
                                                                                                                            range=list(proba_palette.values())))
    # 
    chart3.configure_legend(orient='bottom').save('figures/death_gastro_scatter2.html')

    base = alt.Chart(df_combo)
    chart3 = base.mark_circle().encode(x=alt.X('gastrostomy_pred',axis=alt.Axis(style='log')).title('predicted days to gastrostomy'),
                                                y=alt.Y('death_pred:Q',axis=alt.Axis(style='log')).title('predicted overall survival (days)'),
                                                color=alt.Color('lbl:N').title('Gastrostomy probability'),
                                                column='lbl')
    # 
    chart3.configure_legend(orient='bottom').save('figures/death_gastro_scatter3x.html')

    #### plot ROC AUC
    fpr_th, tpr_th, roc_auc = compute_roc_auc_threshold(df_combo)
    fpr_w, tpr_w, roc_auc = compute_roc_auc_weight(df_combo)

    def make_df_now(fpr,tpr,method):
        df = pd.DataFrame([fpr,tpr],index=['fpr','tpr']).T
        df['method'] = method
        return df

    d = pd.concat([make_df_now(fpr_dg,tpr_dg,'logistic regression'),
                    make_df_now(fpr_th,tpr_th,'threshold'),
                    # make_df_now(fpr_svm,tpr_svm,'svm'),
                    make_df_now(fpr_w,tpr_w,'weight')],axis=0)
    d['method'] = d['method'].replace({'threshold':'earlier event'})
    
    diagonal_line = alt.Chart(pd.DataFrame({'fpr': [0, 1], 'tpr': [0, 1]})).mark_line(color='black',size=.5).encode(x='fpr', y='tpr')
    lines = alt.Chart(d).mark_line().encode(x=alt.X('fpr',axis=alt.Axis(tickCount=3)).title('False Positive Rate'),
                                            y=alt.X('tpr',axis=alt.Axis(tickCount=3)).title('True Positive Rate'),
                                        strokeDash = 'method')
    (lines+diagonal_line).configure_legend(orient='bottom-right',labelFontSize=14, titleFontSize=14).configure_axis(labelFontSize=14,
        titleFontSize=14,grid=False).save('figures/roc_auc_death_gastro.html')

        # bar plot
    # aux['count'] = np.round(aux['count'],2)

    lbl_dict = {'Loss follow up':'Loss follow up', 'Gastrostomy':'Gastrostomy', 'Death':'Overall survival'}
    aux['lbl'] = aux['lbl'].map(lbl_dict)
    base = alt.Chart(aux).encode(y=alt.Y('count').scale(domain=[0, 1]).title('proportion'),
                                        x=alt.X('lbl').title(None).sort(['Gastrostomy', 'Overall survival','Loss follow up' ]))
    bars = base.mark_bar().encode(color=alt.Color('gastrostomy_proba3').title('Gastrostomy probability').scale(domain=list(proba_palette.keys()),
                                                                                                                            range=list(proba_palette.values())),
                                tooltip = ['count','lbl','gastrostomy_proba3'])
    # text = base.mark_text().encode(text='count',y=alt.Y('sum(count)'))
    (bars).configure_legend(disable=False).properties(width=150).save('figures/proportions32.html')
    
    
    # create 
    aux['gastrostomy_proba3'] = aux['gastrostomy_proba3'].map({'Possible':'Gastrostomy\nmore likely','Unlikely':'Death\nmore likely'})
    aux['count'] = np.round(aux['count'],2)
    base = alt.Chart(aux).encode(y=alt.Y('gastrostomy_proba3').title('Predicted'),
                                            x=alt.X('lbl').title('True').sort(['Gastrostomy', 'Overall survival','Loss follow up' ]))
    rect = base.mark_rect().encode(color=alt.Color('count').scale(alt.Scale(scheme='blues')))

    text = base.mark_text(align='center', baseline='middle', dx=0, dy=0,color='white').encode(text='count')

    (rect+text).configure_legend(disable=True).properties(width=300,height=220).configure_text(fontSize=14).configure_axis(labelFontSize=14,titleFontSize=14).save('figures/confusion_matrix32.html')
    
# for cutoff in np.arange(-720,720,90):
#     df_combo['gastrostomy_proba2'] = pd.cut(df_combo['gastrostomy_pred']-df_combo['death_pred'],[-np.inf,cutoff,np.inf], 
#                                                 labels=["Possible", "Unlikely"])
#     g = df_combo.groupby('gastrostomy_proba2')
#     aux2 = (g['lbl'].value_counts()/df_combo['lbl'].value_counts()).reset_index()

#     lbl_dict = {'Loss follow up':'Loss follow up', 'Gastrostomy':'Gastrostomy', 'Death':'Overall survival'}
#     aux2['lbl'] = aux2['lbl'].map(lbl_dict)
#     base = alt.Chart(aux2).encode(y=alt.Y('count').scale(domain=[0, 1]).title('proportion'),
#                                         x=alt.X('lbl').title(None).sort(['Gastrostomy', 'Overall survival','Loss follow up' ]))
#     bars = base.mark_bar().encode(color=alt.Color('gastrostomy_proba2').title('Gastrostomy probability').scale(domain=list(proba_palette.keys()),
#                                                                                                                             range=list(proba_palette.values())),
#                                 tooltip = ['count','lbl','gastrostomy_proba2'])
#     # text = base.mark_text().encode(text='count',y=alt.Y('sum(count)'))
#     (bars).configure_legend(disable=False).properties(width=150).save(f'figures/proportions32_{cutoff}.html')

#     print(cutoff,":",aux2.query('gastrostomy_proba2 == "Possible" and lbl == "Gastrostomy"')['count'].values+aux2.query('gastrostomy_proba2 == "Unlikely" and lbl == "Overall survival"')['count'].values)

    # now plot the probability curves side by side
    dmat = clf_gastro.dmat_builder(Xg,yg)

    clf_gastro.predict_baseline_hazard(pd.DataFrame(yg))
    surv_curve = clf_gastro.predict_survival_function(dmat)

    # dmat = clf_death.dmat_builder(Xd[dffeat['tcens']>0],yd)
    # y_pred_death = clf_death.final_model.predict(dmat)

    # dfcombo is for encals already
    cols = ['gastrostomy_pred', 'death_pred', 
    'gastrostomy_gt','death_gt']
        # base = alt.Chart(df_combo.reset_index()).mark_circle()
        # line1 = base.encode(x='index',y='death_pred',color=alt.value('red'))
        # line2 = base.encode(x='index',y='gastrostomy_pred')
        # (line1+line2).save('aa.html')

    from lifelines import KaplanMeierFitter

    def get_survival_fcn(kmf,name):
        df = kmf.survival_function_
        df['label'] = name
        df = df.reset_index()
        df.columns = ['time','proportion','label']
        return df

    def calculate_KM(T,E,name):
        kmf_ = KaplanMeierFitter()
        kmf_.fit(T, E, label=name)
        return get_survival_fcn(kmf_,name)

    kmf = [calculate_KM(df_combo['gastrostomy_gt'],df_combo['gastro_bool'],'Gastrostomy True')]
    kmf.append(calculate_KM(df_combo['death_gt'],df_combo['death_bool'],'Death True'))
    kmf.append(calculate_KM(df_combo['gastrostomy_pred'],df_combo['gastro_bool'],'Gastrostomy Pred'))
    kmf.append(calculate_KM(df_combo['death_pred'].fillna(0.),df_combo['death_bool'],'Death Pred'))

    kmf = pd.concat(kmf,axis=0)

    kmf['proportion'] = 1 - kmf['proportion']

    palette_gastr = {'Gastrostomy True':'#9BC9E2','Gastrostomy Pred':'#4076B3','Death True':'#db7d83','Death Pred':'#c26ac8'}

    # label_dict = {'Gastrostomy True':'Gastrostomy True', 'Death True', 'Gastrostomy Pred', 'Death Pred'}

    alt.Chart(kmf).mark_line().encode(x='time',y='proportion',
                                    color=alt.Color('label').scale(domain=list(palette_gastr.keys()),
                                                                    range=list(palette_gastr.values()))).configure_legend(orient='bottom').save('figures/km_plot_predb.html')

# from lifelines.utils import median_survival_times
# from lifelines.datasets import load_waltons
# import matplotlib.pyplot as plt
# from lifelines.plotting import add_at_risk_counts

# plt.figure()
# ax = plt.subplot(111)

    # add_at_risk_counts(*kmf,  
    #                 ax=ax,
    #                 rows_to_show=["At risk"])

    # plt.tight_layout()
    # plt.show()


# Simulation 2: Sensitivity analysis - forward propagation sensitivity
# load the data
if 2 in simulations:
    if recompute_flag:
        O = []
        for model in ['XGBoostCox','XGBoostMAEPOReg']:
            for dataset in ['onset_slope']:
                tic = time.time()
                _,Xg,yg,_,strat_var,feature_names = load_data(dataset,model)
                _,Xg2,yg2,_,strat_var2,_ = load_data(dataset,model,suffix='83_2')
                _,Xg4,yg4,_,strat_var4,_ = load_data(dataset,model,suffix='83_4')
                _,Xg6,yg6,_,strat_var6,_ = load_data(dataset,model,suffix='83_6')
                print('load the data:',model,dataset,time.time() - tic)
                
                for database in ['PROACT','ArQ','IDPP']:
                    tic = time.time()
                    # load clf gastro
                    with open(f'models/gastro_{model}_{dataset}_{database}.pkl', 'rb') as fid:
                        clf_gastro = pickle.load(fid)

                    # evaluate on MedianAE of prediction based on increasing timepoints
                    for X,y,n_points,strat_var in [[Xg,yg,'all',strat_var],[Xg2,yg2,'2',strat_var2],[Xg4,yg4,'4',strat_var4],[Xg6,yg6,'6',strat_var6]]:
                        dmat = clf_gastro.dmat_builder(X,y)
                        y_pred_gastro = clf_gastro.final_model.predict(X) if model == 'SkSurvCoxLinear' else clf_gastro.final_model.predict(dmat)
                        test_idx = np.where(strat_var==database)[0]
                        train_idx = np.where(strat_var!=database)[0]

                        dtest = clf_gastro.dmat_builder(X[test_idx,:], y[test_idx])
                        dtrain_valid_combined = clf_gastro.dmat_builder(X[train_idx,:], y[train_idx])

                        # get predictions
                        y_pred = clf_gastro.modelnow.predict(dtest)
                        y_pred_train = clf_gastro.modelnow.predict(dtrain_valid_combined)

                        train_uncensored = np.array([yy[0] for yy in y[train_idx]])
                        test_uncensored = np.array([yy[0] for yy in y[test_idx]])
  
                        outnow = calc_all_metrics(pd.DataFrame(),None,clf_gastro,None,dtrain_valid_combined, dtest,[train_uncensored,test_uncensored])
                        outnow['model'] = model
                        outnow['features'] = dataset
                        outnow['n_points'] = n_points
                        outnow['test_rotation'] = database
                        O.append(outnow)
                    print('compute forward performance data:',model,dataset,database,time.time() - tic)

        out = pd.concat(O,axis=0,ignore_index=True)
        out.to_csv('data/forward_propagation_results83.csv',index=False)
    else:
        out = pd.read_csv('data/forward_propagation_results83.csv')

    selected_cols = {'MedianAE_test_uncensored':'MedianAE','Cindex_test_uncensored':'Cindex'}
    outsimple = out.groupby(['model','features','n_points'])[list(selected_cols.keys())].mean().reset_index(drop=False)
    outsimple = outsimple.melt(id_vars=['model','features','n_points'])
    outsimple['variable'] = outsimple['variable'].map(selected_cols)
        
    base = alt.Chart(outsimple).encode(x='n_points',y=alt.Y('value').title(None),color=alt.Color('model').scale(domain=list(palette.keys()),range=list(palette.values())),
                                    tooltip=['n_points','value','model','features','variable'])
    dots = base.mark_circle()
    lines = base.mark_line().encode(strokeDash='features')
    (dots+lines).facet(column=alt.Column('variable',title=None)).resolve_scale(y='independent').save('figures/forward_prop_sensitivity83.html')

    # Simulation 3: Sensitivity analysis - feature sensitivity - Permutation importance
if 3 in simulations:
    # permutation importance    
    with open('models/gastro_XGBoostMAEPOReg_classes_neqm_ArQ.pkl', 'rb') as fid:
        clf_gastro = pickle.load(fid)
        
    fixed_args = load_data()
    _,Xg,yg,strat_var,feature_names = select_model_and_features('classes_neqm','XGBoostMAEPOReg',fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger,export_feature_names=True)
    if recompute_flag:
        ygt = np.array([yy[1] for yy in yg]).astype(float)
        dmat = clf_gastro.dmat_builder(Xg,yg)
        y_pred_gastro = clf_gastro.final_model.predict(dmat)

        out = pd.DataFrame(cal_metrics(ygt,y_pred_gastro),index=[0])

        n_repeats = 3
        for fi in range(Xg.shape[1]):
            for ri in range(n_repeats):
                Xgnow = Xg.copy()
                np.random.shuffle(Xgnow[:,fi])
                dmat = clf_gastro.dmat_builder(Xgnow,yg)
                y_pred_gastro = clf_gastro.final_model.predict(dmat)
                outnow = pd.DataFrame(cal_metrics(ygt,y_pred_gastro),index=[0])
                outnow['repeat'] = ri
                outnow['feature'] = fi
                out = pd.concat([out,
                        outnow],axis=0,ignore_index=True)
                print(ri,fi)
        out.to_csv('data/permutation_importance63.csv',index=False)
    else:
        out = pd.read_csv('data/permutation_importance63.csv')

    out['feature'] = out['feature'].fillna('all').astype(str)

    out2 = out.groupby('feature').mean().drop(columns=['repeat']).reset_index()
    ref = out2.query('feature == "all"')['MedianAE'].values[0]
    out2['MedianAE_Err'] = np.abs(out2['MedianAE']-ref)/ref
    out2 = out2.sort_values('MedianAE_Err',ascending=False)
    idxs = out2.query("MedianAE_Err>0.05")['feature'].values.astype(float).astype(int)
    print('Top importance features > 5% MedianAE change:',np.array(feature_names)[idxs])

    out3 = out2.query('feature != "all"')
    out3.feature = np.array(feature_names)[out3.feature.astype(float).astype(int)]
    out3 = out3.sort_values('MedianAE_Err',ascending=False)
    out3['significant'] = out3['MedianAE_Err']>0.05
    out3['MedianAE_Err%'] = out3['MedianAE_Err']/out3['MedianAE_Err'].sum()

    base = alt.Chart(out3).encode(y=alt.Y('feature:N').sort('-x'),x='MedianAE_Err%:Q',tooltip=['MedianAE_Err%','MedianAE_Err','feature'])
    (base.mark_bar()+base.mark_text(align='left', dx=2).encode(text=alt.Text('MedianAE_Err%:Q', format='.2f'))).transform_filter(alt.FieldGTPredicate(field='MedianAE_Err',
                                                        gt=0.00)).save('figures/PI.html')
    # ,
    # top features - 'classes+decay','XGBoostMAEPOReg'
    # ['b_days', 'a_% predicted', 'b_% predicted', 'age_at_onset',
    #     'b_ALSFRS_Total', 'bulbar_subscore_4', 'diagnostic_delay_months',
    #     'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS',
    #     'ALSFRS_Slope_Onset_to_FirstALSFRS']


    # Simulation 4: Sensitivity analysis - missingness sensitivity
if 4 in simulations:

# load data, models
    if recompute_flag:
        Out = pd.DataFrame()
        
        # models_df = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/time_prediction/data/results_strat_xgbmaepo_weighted_all6_no_imputation.csv')
        # models_df = models_df.query('is_best_trial == True and dataset in ("demo", "classes_neqm")')
        
        for f in ['onset_slope']:
            for m in ['XGBoostRegPO','XGBoostCox']:
                for db in ['PROACT','ArQ','IDPP']:
                    fixed_args = load_data(f,m)
                    with open(f'models/gastro_{m}_{f}_{db}.pkl', 'rb') as fid:
                        clf_gastro = pickle.load(fid)

                    print('baseline missingness',"{:.2f}".format(np.isnan(fixed_args[1]).sum().sum()/np.size(fixed_args[1])*100),'%')
                    
                    # get random missingness
                    notnans = ~np.isnan(fixed_args[1])
                    for remove_pct in [.1,.2,.3,.5]:
                        for test_fold_id in range(3):
                            # try:
                            if True:
                                Xnow = copy.deepcopy(fixed_args[1])
                                samples_to_remove = notnans.sum(axis=0)*remove_pct//1
                                for col in np.arange(Xnow.shape[1]):
                                    idxs = np.where(notnans[:,col])[0]
                                    np.random.shuffle(idxs)
                                    idxnow = idxs[:int(samples_to_remove[col])]
                                    Xnow[idxnow,col] = np.nan
                                print('new missingness',"{:.2f}".format(np.isnan(Xnow).sum()/fixed_args[1].size*100),'%')
                                
                                # get baseline model/data
                                # _,Xg,yg,strat_var,_ = select_model_and_features(f,m,Xnow,fixed_args[2],fixed_args[3],fixed_args[4])
                                Xg = copy.deepcopy(Xnow)
                                yg = copy.deepcopy(fixed_args[2])
                                strat_var = copy.deepcopy(fixed_args[4])
                                # dataset,method,X0,y00,strat_var0,feature_key,export_feature_names = False
                                
                                # get the train test split
                                # idx_test = (strat_var==db).reshape(-1,)
                                # 
                                # X_test,y_test = Xg[idx_test,:],yg[idx_test]
                                # X_train,y_train = Xg[idx_train,:],yg[idx_train]
                                X_train, X_test, y_train, y_test,idx_test = split_data_add_mice(Xg,yg,strat_var,db)
                                idx_train = ~idx_test
                                dtrain_valid_combined = clf_gastro.dmat_builder(X_train,y_train)
                                dtest = clf_gastro.dmat_builder(X_test,y_test)

                                # calculate baseline
                                # mydict = models_df.query(f'method == "{m}" and test_fold == "{db}" and dataset == "{f}"').to_dict()
                                # mydict_list_values = {k: v.values for k, v in mydict.items()}
                                # params_df = {'learning_rate':0.08,'max_depth': 8,'reg_alpha': 30,'reg_lambda': 5} if m == 'XGBoostMAEPOReg' else {'alpha':0.07}
                                y_pred,y_pred_train = clf_gastro.compute_test_pred(study=None, dtrain_valid_combined=dtrain_valid_combined, dtest=dtest,params_df=clf_gastro.params_df)
                            
                                train_uncensored = np.array([yy[0] for yy in y_train])
                                test_uncensored = np.array([yy[0] for yy in y_test])

                                # eval
                                labels_with_po_train,labels_with_po_test = get_po_times(y[train_valid_idx]),get_po_times(y[test_idx])
                                calc_all_metrics(study,model_obj,test_fold_id,dtrain_valid_combined,dtest,labels_with_po,trials_to_compute=[])
                                out = calc_all_metrics(None,clf_gastro,None,dtrain_valid_combined, dtest)
                                out['model'] = m
                                out['features'] = f
                                out['test_database'] = db
                                out['remove_pct'] = remove_pct
                                out['test_fold_id'] = test_fold_id
                                Out = pd.concat([Out,out],axis=0,ignore_index=True)
                                print(m,f,db,remove_pct,test_fold_id)
            # Out.to_csv('data/missingness_test63.csv',index=False)
                            # except Exception as e:
                            #     print(e)
        Out.to_csv('data/missingness_test111.csv',index=False)
    else:
        Out = pd.read_csv('data/missingness_test83.csv')
    
    Out2 = Out.groupby(['model', 'features',  'remove_pct','test_database']).agg(['mean','std']).reset_index()
    Out2.columns = Out2.columns.map('|'.join).str.strip('|')
    Out2['MedianAE_lb'] = Out2['MedianAE_test_uncensored|mean']-Out2['MedianAE_test_uncensored|std']
    Out2['MedianAE_ub'] = Out2['MedianAE_test_uncensored|mean']+Out2['MedianAE_test_uncensored|std']
    base = alt.Chart(Out2).encode(x=alt.X('remove_pct:N', axis=alt.Axis(format='.0%')).title('% points removed'),
                                y=alt.Y('MedianAE_test_uncensored|mean').title('MedianAE'),color=alt.Color('model').scale(domain=list(palette.keys()),range=list(palette.values())))
    dots = base.mark_circle()
    line = base.mark_line().encode(strokeDash='features')
    errorbar = base.mark_errorbar().encode(y=alt.Y('MedianAE_lb').title('MedianAE'),
                                        y2='MedianAE_ub')
    (line+dots+errorbar).facet(column=alt.Column('test_database', title=None)).save('figures/sensitivity_missingness111.html')


if 6 in simulations: 
    # plot all the individual predictions
    df = pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/time_prediction/data/summary_database_rotation_best_last0_5Final_indivindiv111_aj.csv')

    # select best database
    cols = ['feature','Cindex_ipcw_mean', 'Cindex_ipcw_lb', 'Cindex_ipcw_ub',
        'MedianAEPO_test_mean', 'MedianAEPO_test_lb', 'MedianAEPO_test_ub']
    df = df.loc[:,cols]
    # dfa = df.sort_values(by=['Cindex_ipcw_mean'],ascending=False)
    # select only neq
    # idx = df['feature'].apply(lambda x: x.endswith('_eq'))
    # idx2 = df['feature'].str.contains('_\d+', regex=True)
    # idx = np.any(np.array([idx,idx2]),axis=0)
    # idx[14] = False
    # dfb = df.loc[~idx,:].sort_values(by=['Cindex_mean'],ascending=False)
    dfb = df.sort_values(by=['Cindex_ipcw_mean'],ascending=False)
    dfb = dfb.rename(columns={c:c.split('_')[-1] for c in dfb.keys()})

    aux1 = dfb.iloc[:,:4]
    aux1['variable'] = 'Cindex'
    aux2 = dfb.iloc[:,[0,4,5,6]]
    aux2['variable'] = 'MedianAE'
    dfb = pd.concat([aux1,
                    aux2],
                    axis=0)
    dfb['feature'] = dfb['feature'].str.replace('_',' ')
    
    # take the mean by variable
    # dfb.loc[:,'feature']  = dfb.loc[:,'feature'].apply(lambda x: x[:x.find(' ord cl neqm')]+" decline class")
    dfb = dfb.groupby(['feature','variable']).median().reset_index().sort_values(by=['variable','mean'],ascending=False)
    
    # select only the incumbent features
    vars_to_remove = ['Database PROACT', 'Clinical','site onset Other','Database IDPP','sex Male','Database ArQ']
    idx = dfb['feature'].isin(vars_to_remove)
    # apply(lambda x:x.find('Database')==-1 and x.find('Clinical')==-1)
    dfb = dfb.loc[~idx,:]

    dfb['feature'] = dfb['feature'].replace({'ALSFRS Slope Onset to FirstALSFRS':'ALSFRS Slope Onset',
                            'ALSFRS Slope Onset to FirstBulbar':'Bulbar Slope Onset',
                            r'ALSFRS Slope Onset to First%FEV':'FEV% Slope Onset',
                            'ALSFRS Slope Onset to Firstq3':'Q3 Slope Onset',
                            'ALSFRS Slope Onset to FirstWeight':'Weight Slope Onset',
                            'weight 1':'5% weightloss'})
    aux = dfb.pivot(index=['feature'],columns=['variable'],values=['mean']).reset_index()
    aux['sorting'] = (aux['mean']['Cindex']+1-(aux['mean']['MedianAE'])/500)/2
    aux = aux.sort_values(by=['sorting'],ascending=False)
    sortingby = aux['feature'].values

    var_label = {'Cindex':'Cindex', 'MedianAE':'Margin MedianAE (days)'}

    def get_bars_dots(dfb, variable,no_axis=False):
        y = alt.Y('feature', axis=None) if no_axis else alt.Y('feature')
        dfbnow = dfb.query(f'variable == "{variable}"')
        baseb = alt.Chart(dfbnow).encode(y = y.sort(sortingby).title(None),
                                        color='variable')
        bars = baseb.mark_bar(opacity=.7,cornerRadius=10, height=10).encode(x = alt.X('lb', scale=alt.Scale(domain=[dfbnow['lb'].min(),
                                                                                                                    dfbnow['ub'].max()])).title(None),
                                                                            x2 = alt.X2('ub').title(var_label[variable]))
        dots = baseb.mark_point(opacity=1,size=60).encode(x = alt.X('mean', scale=alt.Scale(domain=[dfbnow['lb'].min(),
                                                                                                    dfbnow['ub'].max()])).title(var_label[variable]),color = alt.value('black'),
                                                    shape = alt.value('stroke'),
                                                    angle=alt.value(90))
        return (bars+dots)

    chart1 = get_bars_dots(dfb, 'Cindex')
    chart2 = get_bars_dots(dfb, 'MedianAE',no_axis=True)
    (chart1 | chart2).resolve_axis(x='independent').resolve_scale(x='independent').configure_legend(disable=True).save('figures/indiv_model_pred_plot111_aj.html')
