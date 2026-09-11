# plot_right_censored.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file plots model results

import pandas as pd
import os
from utils.utils_censor import *
import logging
from datetime import datetime
import altair as alt
from multiprocessing import Pool

reload = True

# Basic configuration
logging.basicConfig(filename=f'logs/AFT_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pred.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

method_mapping = {'XGBoostRegPO':'MAEMargin XGBoost', 
                      'XGBoostCox':'Cox XGBoost', 
                      'SkSurvCoxLinear':'Cox Linear',
                      'weight':'Naive model'}
dataset_mapping = {'onset_slope':'Onset Slope','demo':'Baseline'}
palette_renamed = {method_mapping[p]:v for p,v in palette.items()}
_suffix = '111_margin_aj_1080_2920_mice'

if __name__ == '__main__':
    # computed_models = [pd.read_csv(os.path.join('logs',l)) for l in os.listdir('logs') if l.endswith('.csv') and l.startswith('simul')]
    # computed_models = pd.concat(computed_models) if len(computed_models)>0 else pd.DataFrame([])
    # # get model best results
    # models_df = pd.concat([pd.read_csv(os.path.join('logs',l)) for l in os.listdir('logs') if l.endswith('.csv') and not l.startswith('simul')])
    # models_df.sort_values(by=['Cindex'],ascending=False)
    # models_df.drop_duplicates(['dataset', 'method','kernel'],keep='first',inplace=True)
    # # models_df.loc[models_df.loc[:,'method']=='LinearSVRAFT','kernel'] = 'linear'
    # models_df.reset_index(drop=True,inplace=True)
    # models_df = models_df.query('kernel!=kernel or kernel in ("linear","rbf")').reset_index(drop=True)
    # models_df.to_csv('data/all_results_AFT.csv',index=False)
    
    # using the retraining or inference
    # models_df = pd.read_csv(f'data/summary_database_rotation_best_last0_5Final_newclass{suffix_}.csv')
    models_df = pd.read_csv(f'data/results_best_strat_xgbmaepo_weighted_all{_suffix}.csv')
    
    # using the best results from the hyperparameter optimization
    # models_df = pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/time_prediction/data/results_best_strat_xgbmaepo_weighted_all93_no_imputation.csv')
    # models_df.rename(columns={'test_fold':'test_rotation'},inplace=True)
    
    models_df = models_df.query('dataset != "demo+onset_slope"')
    models_df['method'] = models_df['method'].map(method_mapping)

    # PLOT 1: All variables - higher level API  
    # title_dict = {'':'Censored','_uncensored':'Uncensored'}
    # cols = ['dataset', 'method','test_rotation',
    #             'MedianAEPO_test',
    #             # 'PredIn90_test',
    #             # 'PredIn180_test',
    #             # 'PredIn360_test',
    #             'Cindex_IPCW_true_risk',
    #             'MedianAEPO_train',
    #             # 'PredIn90_train',
    #             # 'PredIn180_train',
    #             # 'PredIn360_train',
    #             # 'Cindex_train'
    #             ]
    # for suffix in ['','_uncensored']:
    #     cols = cols[:3]+[c+suffix for c in cols[3:]]
    #     modelf_df_r = models_df.loc[:,cols].sort_values(by = ['Cindex_test'+suffix],ascending=False).drop_duplicates(subset=['dataset', 'method','test_rotation'],keep='first')
    #     modelf_df_r = modelf_df_r.melt(id_vars=['dataset', 'method','test_rotation'])
    #     modelf_df_r['data_split'] = modelf_df_r['variable'].apply(lambda x:'_'.join(x.split('_')[1:]))
    #     modelf_df_r['data_split_dummy'] = modelf_df_r['data_split'].map({k:' '*ki for ki,k in enumerate(modelf_df_r['data_split'].unique())})
    #     modelf_df_r['variable'] = modelf_df_r['variable'].apply(lambda x:x.split('_')[0])
    #     modelf_df_r['value'] = modelf_df_r['value'].round(2)
    #     modelf_df_r['method'] = modelf_df_r['method']#+modelf_df_r['data_split']
    #     # modelf_df_r.loc[modelf_df_r['data_split']=='train'+suffix,'dataset'] = ''
    #     modelf_df_r.loc[:,'dataset'] += modelf_df_r['data_split_dummy']
    #     modelf_df_r.dropna(subset=['value','dataset'],how='any',inplace=True)
    #     for rot in modelf_df_r['test_rotation'].unique():
    #         model_now = modelf_df_r.query(f'test_rotation == "{rot}"')
    #         base = alt.Chart(model_now).encode(y='dataset',
    #                             x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
    #                                     color=alt.Color('data_split',scale=alt.Scale(domain=['test'+suffix,'train'+suffix],range=['#399918','#9CDBA6'])),
    #                                     text = 'value',
    #                                     detail='data_split',
    #                                     tooltip=['dataset', 'method', 'variable', 'value', 'data_split'])

    #         (base.mark_bar()+base.mark_text(align='left', dx=2)).properties(width=100,height=100).facet(column=alt.Column('variable',sort=['Cindex','MedianAE', 'PredIn90', 'PredIn180', 'PredIn360']),row='method').properties(title=title_dict[suffix]+" for "+ rot).resolve_scale(x='independent').save(f'figures/results_metrics_Surv_Final{suffix}_{rot}{suffix_}.html')
    #         print(f'{suffix}_{rot} Done!')
    #     # make the average
    #     modelf_df_m = modelf_df_r.groupby(['dataset','method','variable','data_split'])['value'].mean().reset_index()
    #     base = alt.Chart(model_now).encode(y=alt.Y('dataset').title(None),
    #                             x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
    #                                     color=alt.Color('data_split',scale=alt.Scale(domain=['test'+suffix,'train'+suffix],range=['#399918','#9CDBA6'])).title('Data Split'),
    #                                     text = alt.Text('value',format=alt.condition(alt.datum.variable!="MedianAE", alt.value('.2f'), alt.value('.0f'))),
    #                                     detail='data_split',
    #                                     tooltip=['dataset', 'method', 'variable', 'value', 'data_split'])

        
    #     for vi,variable in enumerate(['Cindex','MedianAEPO', 'PredIn90', 'PredIn180', 'PredIn360']):
    #         title_options = {'text': variable,'fontSize': 14,'anchor':'middle'}
    #         if vi==0: 
    #             chart = (base.mark_circle()+base.mark_text(align='left', dx=4)).transform_filter(alt.FieldEqualPredicate(field='variable',equal=variable)).properties(width=120,height=120).facet(row=alt.Row('method',title=None)).properties(title=title_options).resolve_scale(x='shared')
    #         else:
    #             chart |= (base.mark_circle()+base.mark_text(align='left', dx=4)).transform_filter(alt.FieldEqualPredicate(field='variable',equal=variable)).properties(width=120,height=120).facet(row=alt.Row('method',title=None)).properties(title=title_options).resolve_scale(x='shared')
    #     chart.configure_axis(labelFontSize=10,titleFontSize=14).configure_legend(orient='bottom',labelFontSize=10,titleFontSize=12).properties(title={'text':title_dict[suffix]+" for Average",
    #                                                                                                                                                                      'fontSize': 14}).save(f'figures/results_metrics_Surv_Final{suffix}_avgNFinal{suffix_}.html')
    # print(f'{suffix} Average Done!')

    # # PLOT 2: All variables - lower level API  
    # cols = ['dataset', 'method','Cindex',
    #     'n_features',
    #     'n_patients']#'kernel',
    # modelf_df_r = models_df.loc[:,cols].sort_values(by = ['Cindex'],ascending=False).drop_duplicates(subset=['dataset', 'method'],keep='first')#,'kernel'
    # modelf_df_r = modelf_df_r.melt(id_vars=['dataset', 'method'])#,'kernel'
    # # modelf_df_r['kernel'].fillna('',inplace=True)
    # # modelf_df_r['method'] += modelf_df_r['kernel']
    # modelf_df_r['value'] = modelf_df_r['value'].round(2)
    # modelf_df_r.dropna(subset=['value'],inplace=True)
    # base = alt.Chart(modelf_df_r)
    # base = base.mark_bar().encode(y='method',
    #                         x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
    #                                 # color=alt.Color('data_split',scale=alt.Scale(domain=['test','train'],range=['#399918','#9CDBA6'])),
    #                                 text = 'value',
    #                                 tooltip=['dataset', 'method', 'variable', 'value'])

    # (base.mark_bar()+base.mark_text(align='left', dx=2)).properties(width=100,height=100).facet(column='variable:N',row='dataset').resolve_scale(x='independent').save('data/results_metrics_Surv_all_Final.html')

    # plot confidence intervals!
    # there is a problem with the calculation - the brackets are not correct
    cols = ['method','dataset','test_fold','Cindex_ipcw_mean',
        'Cindex_ipcw_lb', 'Cindex_ipcw_ub', 'MedianAEPO_test_mean', 'MedianAEPO_test_lb', 'MedianAEPO_test_ub']
    #,    'PredIn90_mean', 'PredIn90_lb', 'PredIn90_ub']
    models_dfnow = models_df.loc[:,cols]
    a = models_dfnow.iloc[:,:6].rename(columns={'Cindex_ipcw_mean':'value','Cindex_ipcw_ub':'ub','Cindex_ipcw_lb':'lb'})
    a['variable'] = 'Cindex_ipcw'
    b = models_dfnow.iloc[:,[0,1,2,6,7,8]].rename(columns={'MedianAEPO_test_mean':'value','MedianAEPO_test_ub':'ub','MedianAEPO_test_lb':'lb'})
    b['variable'] = 'MedianAEPO'
    # c = models_dfnow.iloc[:,[0,1,2,9,10,11]].rename(columns={'PredIn90_mean':'value','PredIn90_ub':'ub','PredIn90_lb':'lb'})
    # c['variable'] = 'PredIn90'
    models_dfnow = pd.concat([a,b],axis=0,ignore_index=True)
    varmap = {'Cindex_ipcw':'Concordance Index', 'MedianAEPO':'Margin MedianAE (days)'}#, 'PredIn90':'% predicted within 90 days'}
    models_dfnow['variable'] = models_dfnow['variable'].map(varmap)

    # load weight model - naive
    dfw = pd.read_csv('data/results_weight_naive_model3.csv').rename(columns={'Unnamed: 0':'variable','mean':'value'})
    selected_cols = ['MedianAEPO','Cindex_ipcw']
    dfw = dfw.query(f'variable in {tuple(selected_cols)}')
    dfw['variable'] = dfw['variable'].map(varmap)
    dfw['method'] = 'Naive model'
    dfw['dataset'] = 'weight'
    dfw.rename(columns={'test_rotation':'test_fold'},inplace=True)

    models_dfnow = pd.concat([models_dfnow,dfw],axis=0,ignore_index=True)
    models_dfnow = models_dfnow.query("variable != '% predicted within 90 days'")

        # summary variables with confidence bracket
    dataset_rename_dict = {'onset_slope':'Onset Slope',
                            'demo':'Baseline',
                            'weight':'10% Weightloss',
                            'demo+onset_slope':'Baseline+Slopes'}
    models_dfnow['dataset'] = models_dfnow['dataset'].map(dataset_rename_dict)
    base = alt.Chart(models_dfnow.drop_duplicates(subset=['method','dataset','test_fold','variable'])).encode(y=alt.Y('dataset').title(None),
                                    color = alt.Color('method').scale(domain=list(palette_renamed.keys()),range=list(palette_renamed.values())),
                                    tooltip=['variable','value','ub','lb','method','dataset'])

    
    # (bars + dots).properties(width=200).facet(column=alt.Column('variable').title(None),row=alt.Row('test_rotation').title(None)).resolve_scale(x='independent').save('figures/confidence_bracketFinal.html')
    for vi,var in enumerate(models_dfnow['variable'].unique()):
        dots = base.mark_circle(opacity=1).encode(x=alt.X('value',scale=alt.Scale(zero=False,type='linear'), axis=alt.Axis(tickCount=4)).title(var))                       
        bars = base.mark_errorbar(opacity=1,ticks=True).encode(x=alt.X('lb',scale=alt.Scale(zero=False,type='linear')).title(var),x2='ub')
    
        plotnow = (bars + dots).properties(width=100).transform_filter(alt.FieldEqualPredicate(field='variable',equal=var
                                                                                    )).facet(row=alt.Row('test_fold').title(None))#.properties(title=var)

        if vi==0:
            # chart = plotnow
            chart2 = plotnow
        else:
            # chart &= plotnow
            chart2 = alt.hconcat(chart2,plotnow)
            
    chart2.configure_title(anchor='middle').configure_legend(orient='bottom').resolve_scale(x='independent').save(f'figures/confidence_bracketFinal{_suffix}.html')


    _suffix = '111_aj'
    # alternative
    results = pd.read_csv(f'data/acc_curve_simulation_last0_5Final_newclass{_suffix}.csv')
    results = results.query('dataset != "demo+onset_slope"')
    results['method'] = results['method'].map(method_mapping)
    results['dataset'] = results['dataset'].map(dataset_mapping)

        # results 
    results = results.groupby(['offset','method','dataset','test_rotation'])['PredInTol'].mean().reset_index()
    alt.Chart(results).mark_line(opacity=.7).encode(x=alt.X('offset').title('Tolerance'),
                                    y=alt.Y('PredInTol').title('Prediction within Tolerance'),
                                    color = 'method',
                                    strokeDash='dataset',
                                    column=alt.Column('test_rotation').title(None),
                                    tooltip=['offset','PredInTol','method','dataset']).save(f'figures/PredIn_curveFinal{_suffix}.html')


    # chart.configure_title(anchor='middle').configure_legend(disable=True).transform_filter(alt.FieldOneOfPredicate(field='method',
    #                                                 oneOf=['weight','MAEPO XGBoost'])).resolve_scale(x='independent').save('figures/confidence_bracketFinalFig32.html')

    # chart.configure_title(anchor='middle').configure_legend(disable=True).transform_filter(alt.FieldOneOfPredicate(field='method',
    #                                                 oneOf=['weight','MAEPO XGBoost'])).transform_filter(alt.FieldOneOfPredicate(field='dataset',
    #                                                 oneOf=('classes','encals','weight'))).resolve_scale(x='independent').save('figures/confidence_bracketFinalFig3Simpler2.html')

    # detail of all predictions and their position relative to the Accuracy 90
    # I need train/test and pred for each model/dataset combination
dfpred = pd.read_csv(f'data/predictions_last0_5Final_newclass{_suffix}.csv')
dfpred['method'] = dfpred['method'].map(method_mapping)

dfpred['time_gt'] = dfpred['time_gt'].astype(str)
aa = dfpred.groupby('pseudoid')['time_gt'].describe()['top'].reset_index()
dfpred = dfpred.merge(aa,on='pseudoid')
dfpred['time_gt'] = dfpred['top']
dfpred['time_gt'] = dfpred['time_gt'].astype(float)
window = 90

dfpred['time_lb'] = dfpred['time_gt'] - window
dfpred['time_ub'] = dfpred['time_gt'] + window
dfpred['label'] = np.abs(dfpred['time_pred']-dfpred['time_gt'])>window
dfpred['label'] = dfpred['label'].map({True:'out',False:'in'})

# dfpred2 = dfpred.melt(id_vars=['pseudoid', 'data_split', 'test_rotation','method', 'dataset'],value_name='time',var_name='data_origin')
dfpred2 = dfpred.groupby(['pseudoid', 'data_split', 'method', 'dataset','label']).agg({'time_lb':'mean',  'time_ub':'mean','time_pred':'mean','time_gt':'mean'}).reset_index()
dfpred2 = dfpred2.query('time_ub <= 1080 and data_split == "test" and dataset in ("demo","onset_slope")')

dfpred2['label'] = dfpred2.apply(lambda x:'1080+' if x.time_pred>1080 else x.label,axis=1) 
dfpred2['time_pred'] = dfpred2['time_pred'].clip(lower=0,upper=1080)
aux3 = dfpred2.copy().loc[:,['pseudoid','time_lb']].rename(columns={'time_lb':'order'})
aux3 = aux3.groupby('pseudoid')['order'].min().reset_index()
dfpred2 = dfpred2.merge(aux3,on='pseudoid')

    # aux = dfpred2.   
def gen_indiv_preds(dfpred2aux):
    
    base = alt.Chart(dfpred2aux).encode(
        alt.X("time_pred:Q",axis=alt.Axis(values=[0,1,2,3])).scale(domain=[0,3]).title('year'),#,labelExpr="'year ' + datum.value"
        alt.Y("pseudoid:N",sort=alt.EncodingSortField(field="order", op="min", order='ascending'))                                
                .axis(offset=0, ticks=False, minExtent=0, domain=False,labels=False)
                .title("Participant")
        )

    line = base.mark_errorbar().encode(
        x = alt.X("time_lb:Q").title('year'),#.scale(domain=[0,1080])
        x2 = alt.X2("time_ub:Q"),
        detail="pseudoid:N",
        opacity=alt.value(.8),
        color = alt.value('lightgray'),
    )
    dots = base.mark_circle().encode(
        detail="pseudoid:N",
        opacity=alt.value(.5),
        tooltip=['pseudoid','time_lb','time_ub','time_pred','label','time_gt'],
        color=alt.Color('label').scale(domain=['within ±90days', 'outside ±90days', '>3years'],
                                        range=['blue','orange','gray']).title('predicted:')#alt.value('red'),
    )
    return (line + dots).properties(height=700,width=150)


# rename
dfpred2.loc[:,'time_lb':'time_gt'] = dfpred2.loc[:,'time_lb':'time_gt']/365
dfpred2['label'] = dfpred2['label'].map({'in':'within ±90days',
                                                    'out':'outside ±90days',
                                                    '1080+':'>3years'})


for dataset in ['demo','onset_slope']:
    dfpred2aux = dfpred2.query(f'dataset == "{dataset}"')
    chart = gen_indiv_preds(dfpred2aux)
    chart.facet(column=alt.Column('method:N',header=alt.Header(labelFontSize=12,title=dataset,titleFontSize=14))).configure_axis(labelFontSize=12,titleFontSize=12
            ).configure_legend(labelFontSize=12,title=None,orient='bottom',direction='vertical').save(f'figures/indiv_preds_{dataset}Final.html')
        

##### this is a test only
#     filtering by death
    # load death
    # dfdeath = pd.read_csv('data/encals_overall_survival_pred_final.csv')
    # dfdeath = dfdeath.rename(columns={'OUT':'pred_time_death',
    #                                   'Unnamed: 0':'pseudoid'})

    # need to check whether pseudoid was true or not!!
    dfdeath = pd.read_csv('data/death_preds_gastro_probs.csv')
    dfdeath['pseudoid'] = np.arange(1,dfdeath.shape[0]+1)
    dfdeath = dfdeath.loc[:,['pseudoid','lbl','gastrostomy_proba']]
    
    dfpred2 = dfpred2.merge(dfdeath, on='pseudoid', how='left')

    mapdatasets = {"demo":'Baseline',"onset_slope":'Onset Slope'}
    dfpred2['dataset'] = dfpred2['dataset'].map(mapdatasets)

    method = 'MAEPO XGBoost'
    aux1 = dfpred2.query(f'method == "{method}" and dataset in ("Baseline","Onset Slope")').groupby(['lbl','gastrostomy_proba','label'])['pseudoid'].nunique()
    aux2 = dfpred2.query(f'method == "{method}" and dataset in ("Baseline","Onset Slope")').groupby(['lbl','gastrostomy_proba'])['pseudoid'].nunique()
    print(aux1/aux2)
    (aux1/aux2).to_csv('data/pred_props_accuracy.csv',index=True)
    for ri,row in dfpred2.loc[:,['lbl','gastrostomy_proba']].drop_duplicates().iterrows():
        dfpred2aux = dfpred2.query(f'method == "{method}" and dataset in ("Baseline","Onset Slope") and gastrostomy_proba == "{row["gastrostomy_proba"]}"')
        chart = gen_indiv_preds(dfpred2aux)
        chart.facet(column=alt.Column('dataset:N',header=alt.Header(labelFontSize=12,title=None,titleFontSize=14))).configure_axis(labelFontSize=12,titleFontSize=12
                    ).configure_legend(labelFontSize=12,orient='top',direction='horizontal').save(f'figures/indiv_preds_{method}_{row["gastrostomy_proba"]}_Final{suffix_}.html')

    # now for all of them for MAEPO
    # plot best case scenario - this needs to be addressed!!!!!!!!!!
    # dfpred2['abs_error'] = np.abs(dfpred2['time_gt']-dfpred2['time_pred'])
    # dfpred2 = dfpred2.sort_values(by='abs_error',ascending=True).drop_duplicates(subset=['pseudoid','method','dataset'])

    chart = gen_indiv_preds(dfpred2.query('method=="MAEMargin XGBoost"'))
    chart.facet(column=alt.Column('dataset:N',header=alt.Header(labelFontSize=12,title=None,titleFontSize=14))).configure_axis(labelFontSize=12,titleFontSize=12
                    ).configure_legend(labelFontSize=12,orient='top',direction='horizontal').save(f'figures/indiv_preds_{method}_Final{_suffix}.html')

    # % in
    dfpred2['year'] = pd.cut(dfpred2['time_gt'],bins=[0,1,2,3,np.inf],labels=['1','2','3','3+'])
    aux = dfpred2.query('method == "MAEPO XGBoost" and label!="1080+" and dataset in ("Baseline","Onset Slope")')
    all = aux.groupby(['dataset','year'])['pseudoid'].nunique()
    inonly = aux.query('label == "within ±90days"').groupby(['dataset','year'])['pseudoid'].nunique()
    print(inonly/all)

    # distributions
    interval = 1/12
    bins = np.arange(0,3+interval/2,interval)
    aux['time_gt_discr'] = pd.cut(aux['time_gt'],bins=bins,labels=bins[1:]-interval/2)
    aux['pseudoid'] = aux['pseudoid'].astype(str)
    # this stopped working...
    aux2 = aux.groupby(['time_gt_discr','dataset','label'])['pseudoid'].count().astype(float)
    
    #### this does not work anymore
    aux2 = pd.DataFrame([list(l[0])+[l[1].nunique()] for l in aux.groupby(['time_gt_discr','dataset','label'])['pseudoid']],
                        columns=['time_gt_discr','dataset','label','pseudoid'])
    aux2['pseudoid'] = aux2['pseudoid'].astype(float)
    all = aux2.groupby(['time_gt_discr','dataset'])['pseudoid'].sum().astype(float).reset_index()
    all = aux2.merge(all,on=['time_gt_discr','dataset'],how='left')
        # aux2 = (aux2.iloc[aux2.index.get_level_values('label') == "within ±90days"]/all).reset_index().drop(columns=['label'])
    aux2['pseudoid'] = (all['pseudoid_x']/all['pseudoid_y'])
    aux2['pseudoid']*=100

    alt.Chart(aux2).mark_bar(opacity=.5).encode(x=alt.X('time_gt_discr:Q',axis=alt.Axis(values=[0,1,2,3],labels=False)).title(None),
                                            y=alt.Y('pseudoid',axis=alt.Axis(tickCount=3,labelExpr='datum.value + "%"')).scale(domain=[0,100]).title(None),
                                            color=alt.Color('label').scale(domain=['within ±90days', 'outside ±90days', '>3years'],
                                            range=['blue','orange','gray']),
                                            tooltip=['time_gt_discr', 'dataset', 'label', 'pseudoid']
                                            ).properties(height=80,width=150).facet(alt.Column('dataset',header=alt.Header(labelFontSize=12,title=None,labels=False))).configure_axis(labelFontSize=12,titleFontSize=12
                    ).configure_legend(disable=True).save(f'figures/indiv_preds_{method}Totals{suffix_}.html')

    aux2['year'] = aux2['time_gt_discr'].astype(int)+1
    print(aux2.query('label == "within ±90days"').groupby(['dataset','year'])['pseudoid'].mean())

    # plot the distribution of the median absolute error
    aux['abs_error'] = np.abs(aux['time_pred'] - aux['time_gt'])
    aux['time_gt_discr'] = pd.cut(aux['time_gt'],bins=bins,labels=bins[1:]-interval/2).astype(float)
    aux3 = aux.groupby(['time_gt_discr','dataset'])['abs_error'].mean().reset_index()
    alt.Chart(aux3).mark_bar(opacity=.5).encode(x=alt.X('time_gt_discr:Q',axis=alt.Axis(values=[0,1,2,3])).scale(domain=[0,3]).title('year'),
                                            y=alt.Y('abs_error').title(None)
                                            ).properties(height=80,width=150).facet(row=alt.Row('dataset',header=alt.Header(labelFontSize=12,title='Pseudo-observed MAE (years)'))).configure_axis(labelFontSize=12,titleFontSize=12
                    ).configure_legend(disable=True).save(f'figures/indiv_preds_{method}MedianAE{suffix_}.html')

    ##### predictions long versus short
    dfpred = dfpred.rename(columns={c:c+'_test' for c in ['time_pred','time_gt']})
    dfpred['err_test'] = dfpred['time_pred_test'] - dfpred['time_gt_test']
    dfpred['relerr_test'] = dfpred['err_test']/dfpred['time_gt_test']
    dfpred['aberr_test'] = dfpred['time_pred_test'] - dfpred['time_gt_test']
    dfpred['relaberr_test'] = np.abs(dfpred['time_pred_test'] - dfpred['time_gt_test'])/dfpred['time_gt_test']

    dfnow = dfpred.dropna().groupby(['time_gt_test', 'test_rotation','method','dataset'])['err_test'].mean().reset_index()

    alt.Chart(dfnow).mark_circle().encode(x='time_gt_test',y='err_test',
                                        color='method',strokeDash='dataset',column='test_rotation').save(f'figures/linerror_over_timeFinal{suffix_}.html')