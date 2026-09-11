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

method_mapping = {'XGBoostMAEPOReg':'MAEPO XGBoost', 
                      'XGBoostCox':'Cox XGBoost', 
                      'SkSurvCoxLinear':'Cox Linear',
                      'weight':'weight'}
palette_renamed = {method_mapping[p]:v for p,v in palette.items()}


if __name__ == '__main__':
    models_df = pd.read_csv('data/summary_database_rotation_best_last0_5Final.csv')
    models_df['method'] = models_df['method'].map(method_mapping)

    # PLOT 1: All variables - higher level API  
    title_dict = {'':'Censored','_uncensored':'Uncensored'}
    cols = ['dataset', 'method','test_rotation',
                'MedianAE_test',
                'PredIn90_test',
                'PredIn180_test',
                'PredIn360_test',
                'Cindex_test',
                'MedianAE_train',
                'PredIn90_train',
                'PredIn180_train',
                'PredIn360_train',
                'Cindex_train']
    for suffix in ['','_uncensored']:
        cols = cols[:3]+[c+suffix for c in cols[3:]]
        modelf_df_r = models_df.loc[:,cols].sort_values(by = ['Cindex_test'+suffix],ascending=False).drop_duplicates(subset=['dataset', 'method','test_rotation'],keep='first')
        modelf_df_r = modelf_df_r.melt(id_vars=['dataset', 'method','test_rotation'])
        modelf_df_r['data_split'] = modelf_df_r['variable'].apply(lambda x:'_'.join(x.split('_')[1:]))
        modelf_df_r['data_split_dummy'] = modelf_df_r['data_split'].map({k:' '*ki for ki,k in enumerate(modelf_df_r['data_split'].unique())})
        modelf_df_r['variable'] = modelf_df_r['variable'].apply(lambda x:x.split('_')[0])
        modelf_df_r['value'] = modelf_df_r['value'].round(2)
        modelf_df_r['method'] = modelf_df_r['method']#+modelf_df_r['data_split']
        # modelf_df_r.loc[modelf_df_r['data_split']=='train'+suffix,'dataset'] = ''
        modelf_df_r.loc[:,'dataset'] += modelf_df_r['data_split_dummy']
        modelf_df_r.dropna(subset=['value','dataset'],how='any',inplace=True)
        for rot in modelf_df_r['test_rotation'].unique():
            model_now = modelf_df_r.query(f'test_rotation == "{rot}"')
            base = alt.Chart(model_now).encode(y='dataset',
                                x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
                                        color=alt.Color('data_split',scale=alt.Scale(domain=['test'+suffix,'train'+suffix],range=['#399918','#9CDBA6'])),
                                        text = 'value',
                                        detail='data_split',
                                        tooltip=['dataset', 'method', 'variable', 'value', 'data_split'])

            (base.mark_bar()+base.mark_text(align='left', dx=2)).properties(width=100,height=100).facet(column=alt.Column('variable',sort=['Cindex','MedianAE', 'PredIn90', 'PredIn180', 'PredIn360']),row='method').properties(title=title_dict[suffix]+" for "+ rot).resolve_scale(x='independent').save(f'figures/results_metrics_Surv_Final{suffix}_{rot}.html')
            print(f'{suffix}_{rot} Done!')
        # make the average
        modelf_df_m = modelf_df_r.groupby(['dataset','method','variable','data_split'])['value'].mean().reset_index()
        base = alt.Chart(model_now).encode(y=alt.Y('dataset').title(None),
                                x=alt.X('value', axis=alt.Axis(labels=False)).title(None),
                                        color=alt.Color('data_split',scale=alt.Scale(domain=['test'+suffix,'train'+suffix],range=['#399918','#9CDBA6'])).title('Data Split'),
                                        text = alt.Text('value',format=alt.condition(alt.datum.variable!="MedianAE", alt.value('.2f'), alt.value('.0f'))),
                                        detail='data_split',
                                        tooltip=['dataset', 'method', 'variable', 'value', 'data_split'])

        
        for vi,variable in enumerate(['Cindex','MedianAE', 'PredIn90', 'PredIn180', 'PredIn360']):
            title_options = {'text': variable,'fontSize': 14,'anchor':'middle'}
            if vi==0: 
                chart = (base.mark_circle()+base.mark_text(align='left', dx=4)).transform_filter(alt.FieldEqualPredicate(field='variable',equal=variable)).properties(width=120,height=120).facet(row=alt.Row('method',title=None)).properties(title=title_options).resolve_scale(x='shared')
            else:
                chart |= (base.mark_circle()+base.mark_text(align='left', dx=4)).transform_filter(alt.FieldEqualPredicate(field='variable',equal=variable)).properties(width=120,height=120).facet(row=alt.Row('method',title=None)).properties(title=title_options).resolve_scale(x='shared')
        chart.configure_axis(labelFontSize=10,titleFontSize=14).configure_legend(orient='bottom',labelFontSize=10,titleFontSize=12).properties(title={'text':title_dict[suffix]+" for Average",
                                                                                                                                                                         'fontSize': 14}).save(f'figures/results_metrics_Surv_Final{suffix}_avgNFinal.html')
    print(f'{suffix} Average Done!')

    # # PLOT 2: All variables - lower level API  

    results = pd.read_csv('data/acc_curve_simulation_last0_5Final.csv')
    results['method'] = results['method'].map(method_mapping)

    # results 
    alt.Chart(results).mark_line(opacity=.7).encode(x=alt.X('offset').title('Tolerance'),
                                    y=alt.Y('PredInTol').title('Prediction within Tolerance'),
                                    color = 'method',
                                    strokeDash='dataset',
                                    column=alt.Column('test_rotation').title(None),
                                    tooltip=['offset','PredInTol','method','dataset']).save('figures/PredIn_curveFinal.html')

    # plot confidence intervals!
    # there is a problem with the calculation - the brackets are not correct
    cols = ['method','dataset','test_rotation','Cindex_mean',
        'Cindex_lb', 'Cindex_ub', 'MedianAE_mean', 'MedianAE_lb', 'MedianAE_ub',
        'PredIn90_mean', 'PredIn90_lb', 'PredIn90_ub']
    models_dfnow = models_df.loc[:,cols]
    a = models_dfnow.iloc[:,:6].rename(columns={'Cindex_mean':'value','Cindex_ub':'ub','Cindex_lb':'lb'})
    a['variable'] = 'Cindex'
    b = models_dfnow.iloc[:,[0,1,2,6,7,8]].rename(columns={'MedianAE_mean':'value','MedianAE_ub':'ub','MedianAE_lb':'lb'})
    b['variable'] = 'MedianAE'
    c = models_dfnow.iloc[:,[0,1,2,9,10,11]].rename(columns={'PredIn90_mean':'value','PredIn90_ub':'ub','PredIn90_lb':'lb'})
    c['variable'] = 'PredIn90'
    models_dfnow = pd.concat([a,b,c],axis=0,ignore_index=True)
    varmap = {'Cindex':'Concordance Index', 'MedianAE':'Median Absolute Error (days)', 'PredIn90':'% predicted within 90 days'}
    models_dfnow['variable'] = models_dfnow['variable'].map(varmap)

    # load weight model - naive
    dfw = pd.read_csv('data/results_weight_naive_model.csv').rename(columns={'Unnamed: 0':'variable','mean':'value'})
    selected_cols = ['MedianAE','PredIn90','Cindex']
    dfw = dfw.query(f'variable in {tuple(selected_cols)}')
    dfw['variable'] = dfw['variable'].map(varmap)
    dfw['method'] = dfw['dataset'] = 'weight'

    models_dfnow = pd.concat([models_dfnow,dfw],axis=0,ignore_index=True)

    # summary variables with confidence bracket
    base = alt.Chart(models_dfnow.drop_duplicates(subset=['method','dataset','test_rotation','variable'])).encode(y=alt.Y('dataset').title(None),
                                    color = alt.Color('method').scale(domain=list(palette_renamed.keys()),range=list(palette_renamed.values())),
                                    tooltip=['variable','value','ub','lb','method','dataset'])
    dots = base.mark_circle(opacity=1).encode(x=alt.X('value',scale=alt.Scale(zero=False)).title(None))
                                    
    bars = base.mark_errorbar(opacity=1,ticks=True).encode(
                                    x=alt.X('lb',scale=alt.Scale(zero=False)).title(None),
                                    x2='ub')

    # (bars + dots).properties(width=200).facet(column=alt.Column('variable').title(None),row=alt.Row('test_rotation').title(None)).resolve_scale(x='independent').save('figures/confidence_bracketFinal.html')
    for vi,var in enumerate(models_dfnow['variable'].unique()):
        plotnow = (bars + dots).properties(width=100).transform_filter(alt.FieldEqualPredicate(field='variable',equal=var
                                                                                    )).facet(row=alt.Row('test_rotation').title(None)).properties(title=var)
        if vi==0:
            chart = plotnow
            chart2 = plotnow
        else:
            chart &= plotnow
            chart2 |= plotnow
            
    chart2.configure_title(anchor='middle').configure_legend(orient='bottom').resolve_scale(x='independent').save('figures/confidence_bracketFinal2.html')

    chart.configure_title(anchor='middle').configure_legend(disable=True).transform_filter(alt.FieldOneOfPredicate(field='method',
                                                    oneOf=['weight','MAEPO XGBoost'])).resolve_scale(x='independent').save('figures/confidence_bracketFinalFig32.html')

    chart.configure_title(anchor='middle').configure_legend(disable=True).transform_filter(alt.FieldOneOfPredicate(field='method',
                                                    oneOf=['weight','MAEPO XGBoost'])).transform_filter(alt.FieldOneOfPredicate(field='dataset',
                                                    oneOf=('classes','encals','weight'))).resolve_scale(x='independent').save('figures/confidence_bracketFinalFig3Simpler2.html')

    # detail of all predictions and their position relative to the Accuracy 90
    # I need train/test and pred for each model/dataset combination
    dfpred = pd.read_csv('data/predictions_last0_5Final.csv')
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
    dfpred2 = dfpred2.query('time_ub <= 1080 and data_split == "test" and dataset in ("encals","classes+decay","classes")')

    dfpred2['label'] = dfpred2.apply(lambda x:'1080+' if x.time_pred>1080 else x.label,axis=1) 
    dfpred2['time_pred'] = dfpred2['time_pred'].clip(lower=0,upper=1080)
    aux3 = dfpred2.copy().loc[:,['pseudoid','time_lb']].rename(columns={'time_lb':'order'})
    aux3 = aux3.groupby('pseudoid')['order'].min().reset_index()
    dfpred2 = dfpred2.merge(aux3,on='pseudoid')
        
        # aux = dfpred2.
    def gen_indiv_preds(dfpred2aux):
        base = alt.Chart(dfpred2aux).encode(
            alt.X("time_pred:Q").scale(domain=[0,1080]).title('days'),
            alt.Y("pseudoid:N",sort=alt.EncodingSortField(field="order", op="min", order='ascending'))                                
                    .axis(offset=0, ticks=False, minExtent=0, domain=False,labels=False)
                    .title("Participant"),
                    tooltip=['pseudoid','time_lb','time_ub','time_pred','order']
            )

        line = base.mark_errorbar().encode(
            x = alt.X("time_lb:Q").title('days'),#.scale(domain=[0,1080])
            x2 = alt.X2("time_ub:Q"),
            detail="pseudoid:N",
            opacity=alt.value(.8),
            color = alt.value('lightgray'),
        )
        dots = base.mark_circle().encode(
            detail="pseudoid:N",
            opacity=alt.value(.5),
            color=alt.Color('label').scale(domain=['in', 'out', '1080+'],range=['blue','orange','gray'])#alt.value('red'),
        )
        return (line + dots).properties(height=700,width=150)


    for dataset in ['classes','classes+decay', 'encals']:
        dfpred2aux = dfpred2.query(f'dataset == "{dataset}"')
        chart = gen_indiv_preds(dfpred2aux)
        chart.facet(column=alt.Column('method:N').title(dataset)).configure_legend(orient='bottom').save(f'figures/indiv_preds_{dataset}Final.html')
        
    method = 'MAEPO XGBoost'
    dfpred2aux = dfpred2.query(f'method == "{method}" and dataset in ("encals","classes")')
    chart = gen_indiv_preds(dfpred2aux)
    chart.facet(column=alt.Column('dataset:N').title(method)).configure_legend(orient='bottom').save(f'figures/indiv_preds_{method}Final.html')

    # % in
    dfpred2['year'] = pd.cut(dfpred2['time_gt'],bins=[0,365,730,1080,np.inf],labels=['1','2','3','3+'])
    aux = dfpred2.query('method == "MAEPO XGBoost" and label!="1080+" and dataset in ("encals","classes")')
    all = aux.groupby(['dataset','year'])['pseudoid'].nunique()
    inonly = aux.query('label == "in"').groupby(['dataset','year'])['pseudoid'].nunique()
    print(inonly/all)

    ##### predictions long versus short
    dfpred['err_test'] = dfpred['time_pred_test'] - dfpred['time_gt_test']
    dfpred['relerr_test'] = dfpred['err_test']/dfpred['time_gt_test']
    dfpred['aberr_test'] = dfpred['time_pred_test'] - dfpred['time_gt_test']
    dfpred['relaberr_test'] = np.abs(dfpred['time_pred_test'] - dfpred['time_gt_test'])/dfpred['time_gt_test']

    dfnow = dfpred.dropna().groupby(['time_gt_test', 'test_rotation','method','dataset'])['err_test'].mean().reset_index()

    alt.Chart(dfnow).mark_circle().encode(x='time_gt_test',y='err_test',
                                        color='method',strokeDash='dataset',column='test_rotation').save('figures/linerror_over_timeFinal.html')