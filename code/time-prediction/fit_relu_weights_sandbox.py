# create base visualisation
import os
import sys
sys.path.append('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/app')
import pandas as pd 
import pandas_gbq as pdg
from utils.constants import *
# from utils.utils import *
import numpy as np
import altair as alt
from scipy.optimize import curve_fit
import re

############## this is the final
fulldatapath = '/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data'
df_master = pd.read_csv(fulldatapath+'/master_final_0807.csv',encoding='latin_1',low_memory=False)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jdelgad1/Desktop/Juan/code/.creds/creds-als.json"

# read data
# df = pd.read_csv('data/ALS.TWeight_classes.csv')
df = pd.read_csv('/Users/jdelgad1/Desktop/Juan/code/imperial/imperial-als/data/TALSFRS_classes.csv')

# verticalise the data
poss_id_vars = ['Database', 'numid','tcens', 'variable']

def find_time_col(string,name):
    pattern = f'^d\d+_{name}$'
    return bool(re.match(pattern, string))

# verticalise variables
D = []
suffixes = ['Weight','% predicted', 'q3',  'ALSFRS_Total', 'bulbar_subscore']
for suffix in suffixes:

    new_col_names = {c:c.split('_')[0][1:] for c in df_master.keys() if find_time_col(c,suffix) and suffix in c}
    if suffix == 'Weight' and 'Baseline_Weight' in df_master.keys():
        new_col_names.update({'Baseline_Weight':'0'})
    # new_col_names.update({'id':'numid'})

    df_master.rename(columns={'id':'numid'},inplace=True)
    idvarsnow = [c for c in df_master.keys() if c in poss_id_vars]
    df_masternow = pd.melt(df_master,id_vars=idvarsnow,
            value_vars=list(new_col_names.keys()),
            var_name='days_from_onset',value_name='value')
    df_masternow['days_from_onset'] = df_masternow['days_from_onset'].map(new_col_names)
    df_masternow['variable'] = suffix
    D.append(df_masternow)
D = pd.concat(D,axis=0)
D['days_from_onset'] = D['days_from_onset'].str.replace('p','').astype(int)

D.dropna(subset='value').groupby('variable')['numid'].nunique()

# filter just the weight
df = D.query('variable == "Weight"').loc[:,['numid','days_from_onset','value']].rename(columns={'value':'Weight'})

# calculate the days from onset
dfaux = df.sort_values('days_from_onset').dropna().drop_duplicates(subset=['numid'],keep='first').query('days_from_onset < 180')
dfaux = dfaux.loc[:,['numid', 'Weight']].rename(columns={'Weight':'w0'})
df = df.merge(dfaux,on=['numid'])
df['DeltaW'] = df['Weight'] - df['w0']
df['RelDeltaW'] = (df['DeltaW'])/df['w0']
df['RelDeltaW'].describe()
# df.to_csv('data/weight_verticalised.csv',index=False)

# estimate the delay
def estimate_delay(df,threshold = .05):
    delay = {}
    for gi,g in list(df.sort_values(by=['numid','days_from_onset']).groupby('numid')):
        idx = np.where(g['RelDeltaW']<-threshold)[0]
        if len(idx)>0:
            delay.update({gi:[g.iloc[idx[0],2],1]})
        else:
            delay.update({gi:[g['days_from_onset'].max(),0]})
    
    dfout = pd.DataFrame.from_dict(delay, orient='index', columns=['delay', 'threshold_reached']).reset_index().rename(columns={'index':'numid'})
    dfout['threshold'] = threshold
    return dfout

df.plot(x='days_from_onset',y='RelDeltaW',kind='scatter',alpha=.1)

delay05 = estimate_delay(df,threshold = .05)
delay10 = estimate_delay(df,threshold = .1)

# convert delay dictionary to dataframe
dfout = pd.concat([delay05,delay10],axis=0).merge(df_master.loc[:,['Database','numid','tcens','cens','Dead']],on='numid')
dfout['event_code'] = 2*dfout['Dead'].fillna(0)+(dfout['cens']==0).astype(float)
dfout.loc[dfout.loc[:,'event_code']>2,'event_code'] = 1
dfout['event_code'] = dfout['event_code'].astype(int)
dfout.to_csv(fulldatapath+'/weight_delay_dataset2.csv',index=False)

# pd.concat([delay05,delay10],axis=0).merge(df_master.loc[:,['numid','tcens','cens','Dead']],on='numid').to_csv(fulldatapath+'/weight_delay_dataset_cens.csv',index=False)

# plot them
dfx = df.merge(delay05,on='numid').groupby(['days_from_onset','threshold_reached','numid'])['DeltaW'].mean().reset_index()
selected_numid = np.random.choice(dfx['numid'].unique(), size=1000, replace=False)
dfx['in'] = dfx['numid'].apply(lambda x:x in selected_numid)
dfx = dfx.loc[dfx['in'],:]
dfx['threshold_reached'] = dfx['threshold_reached'].map({0:'No',1:'Yes'})
alt.Chart(dfx).mark_line().encode(x=alt.X('days_from_onset:Q').title('Days'),
                                         y=alt.Y('DeltaW:Q').title('Weight Change from baseline (Kg)'),
                                         detail='numid',
                                         color=alt.Color('threshold_reached:N').scale(domain=['No','Yes'],range=['gray','orange']).title('5% Weight loss reached')).save('test_templates/weightlosstrajectories.html')

print('Weightloss values in Kg by group')
print(df.merge(delay05,on='numid').groupby(['threshold_reached'])['DeltaW'].describe().reset_index())

# aux = pd.concat([delay05,delay10],axis=0)
# aux = aux.query('threshold_reached == 1').groupby('threshold')['delay'].describe().reset_index()
# base = alt.Chart(aux).encode(y='threshold')
# dots = base.mark_circle(x='50%')
# bars = base.mark_bar(x='25%',
#                      x2='75%')
# (dots+bars).save('aaab.html')


df.drop(columns=['w0','DeltaW'],inplace=True)
df_out = df.merge(delay05,on='numid').rename(columns={'threshold_reached':'cl2'})
df_out = df_out.drop_duplicates().dropna(subset=['RelDeltaW'])
df_out = df_out.query('RelDeltaW<.5').reset_index(drop=True)
aa = df_out.groupby(['days_from_onset','cl2'])['RelDeltaW'].describe().reset_index()
aa['lowCI'] = aa['mean']-1.96*aa['std']/np.sqrt(aa['count'])
aa['highCI'] = aa['mean']+1.96*aa['std']/np.sqrt(aa['count'])
aa.to_csv(fulldatapath+'/weight_decay_dataset_cens.csv',index=False)

# df_out.to_csv('data/ALS.TWeight_classes_enriched.csv',index=False)
# pdg.to_gbq(df_out,'ALS.TWeight_classes',project_id='imperial-410612',if_exists='replace')

# now fit the equations
fcn = lambda t,slope,w0,delay: np.clip(w0 - slope * (t - delay),-np.inf,w0)

slopes,covar = {},{}
for numid in df['numid'].unique():
    try:
        print(numid)
        aux = df.query(f'numid == {numid}')
        x = aux['days_from_onset'].values
        y = aux['Weight'].values
        w0 = aux['w0'].unique()[0]
        delay0 = delay[numid]
        custom_fcn = lambda t,slope: fcn(t,slope, w0, delay0)
        
        slope, cov = curve_fit(custom_fcn, x, y,p0=[.5],method='trf')
        slopes.update({numid:slope[0]})
        covar.update({numid:cov[0][0]})
    except:
        print(numid, 'failed')

slopes_df = pd.DataFrame(slopes.items(), columns=['numid', 'slope'])
covar_df = pd.DataFrame(covar.items(), columns=['numid', 'covar'])
delay_df = pd.DataFrame(delay.items(), columns=['numid', 'delay'])

dfout = delay_df.merge(dfaux,on='numid').merge(slopes_df,on='numid').merge(covar_df,on='numid')

dfout.to_csv('data/weight_slope_delay2.csv',index=False)

# np.min(w0 - slope * (x - delay),w0)



# dfout = pd.read_csv('data/weight_slope_delay.csv')

# plot
chart1 = alt.Chart(dfout).mark_bar(opacity=.7).encode(x=alt.X('delay', bin=alt.Bin(step=100)), 
                                y=alt.Y('count()')).properties(title='delay')
chart2 = alt.Chart(dfout).mark_bar(opacity=.7).encode(x=alt.X('w0', bin=alt.Bin(step=10)), 
                                y=alt.Y('count()')).properties(title='weight at onset')
chart3 = alt.Chart(dfout).mark_bar(opacity=.7).encode(x=alt.X('slope', bin=alt.Bin(step=0.05)), 
                                y=alt.Y('count()')).properties(title='slope')
chart = chart1 | chart2 | chart3
chart.save('test_templates/weight_delay2.html')


# get all the delays
dfx = pd.read_csv('data/Design_pred_out_PredictorsAndTcens_Noduplicates.csv')
dfx.rename(columns={'Baseline_Weight':'d0_Weight'},inplace=True)

varcols = [k for k in dfx.keys() if k.startswith('d')]#dfx.loc[:,'d0_ALSFRS_Date':'d0_Weight'].columns
dfxx = dfx.melt(id_vars=[ 'id_num','Dead', 'Death_Date', 'Onset_Site', 'Phenotype',
                    'tcens','cens',
                  'ALSFRS_Slope_Onset_to_FirstALSFRS', 'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS'],
                  value_vars=varcols)
# split column variable into two columns
aux = dfxx['variable'].str.split('_',expand=True).rename(columns={0:'days',1:'varname',2:'type',3:'unit'})
dfxx = pd.concat([dfxx,aux],axis=1)
dfxx['days'] = dfxx['days'].str.replace('d','').str.replace('p','')
dfxx = dfxx.dropna(subset=['value']).reset_index(drop=True)
dfxx['variable'] = (dfxx['varname'] + '_' + dfxx['type'].fillna('') + '_' + dfxx['unit'].fillna('')).str.rstrip('_')
dfxx.drop(columns=['varname','type','unit'],inplace=True)
# dfxx['variable'].unique()

# dfxx.query('variable == "Weight_Loss_pct"')['value'].describe()
# count    12358.000000
# mean         3.034760
# std         32.854032
# min       -939.759036
# 25%          0.000000
# 50%          2.520451
# 75%          7.547170
# max         90.886076
weights = dfxx.query('variable == "Weight"').loc[:,['id_num','days','value']]
aux = weights.groupby('id_num')['value'].first().reset_index().rename(columns={'value':'w0'})
weights = weights.merge(aux,on='id_num')
weights['DeltaW'] = weights['value'] - weights['w0']
weights['RelDeltaW'] = (weights['DeltaW'])/weights['w0']
# weights['RelDeltaW'].describe()

delay = {}
for gi,g in list(weights.sort_values(by=['id_num','days']).groupby('id_num')):
    idx = np.where(g['RelDeltaW']<-threshold)[0]
    if len(idx)>0:
        delay.update({gi:[g.iloc[idx[0],2],True]})
    else:
        delay.update({gi:[g['days'].max(),False]})

# convert delay dictionary to dataframe
delay_df = pd.DataFrame.from_dict(delay, orient='index', columns=['delay', 'exceeded_threshold']).reset_index().rename(columns={'index':'numid'})
delay_df['delay'] = delay_df['delay'].astype(int)
delay_df.to_csv('data/delay_dataset_all.csv',index=False)