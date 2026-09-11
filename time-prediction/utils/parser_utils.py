
import os
import sys
sys.path.append('app')
import pandas as pd 
import pandas_gbq as pdg
import json
from utils.constants import *
from utils.utils import *
import numpy as np
from scipy.optimize import curve_fit
import re

# read credentials
# with open("../creds/sharepoint_creds.json") as fid:
#     creds = json.load(fid)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/juandelgado/Desktop/Juan/code/imperial/creds/google_credentials_als.json"


def parse_classes_data(D,id,varname,maxval,add0,qtrend,save_flag,classvar='cl2',return_df=True):
    first_time = [k for k in D[id].keys() if re.match('t\d+',k)][0]
    df = D[id].melt(id_vars=["numid",classvar], 
                value_vars=D[id].loc[:,first_time:], 
                var_name="days_from_onset", 
                value_name=varname, 
                ignore_index=True)

    df["days_from_onset"] = df["days_from_onset"].str.replace("t","").astype(int)

    # add 48 at t=0
    df = df.dropna(subset=[varname]).reset_index(drop=True)
    if add0:
        aux = df.drop_duplicates(subset=["numid",  classvar]).reset_index(drop=True)
        aux["days_from_onset"] = 0
        aux[varname] = maxval
        df = pd.concat([df,aux],ignore_index=True,axis=0)

    # calculate trend line
    # dark blue (class 2) is  Total ALSFRS=0.0039t^2-0.8788t+48.443  and for 
    # light blue (class 1) is  Total ALSFRS=0.0349t^2-2.2464t+47.324 
    if qtrend:
        df["trendline"] = df.apply(lambda row:fncs[row.cl2](row.days_from_onset/30),axis=1)

        polyfit = {}
        for li,l in list(df.groupby(classvar)):
            l = l.dropna(subset=[varname])
            x = l["days_from_onset"]
            y = l[varname]
            # Fit a polynomial of degree 2 (quadratic)
            degree = 2
            coefficients = np.polyfit(x, y, degree)

            # Generate polynomial function based on the coefficients
            poly_function = np.poly1d(coefficients)
            polyfit.update({li:poly_function})

        df["trendline_juan"] = df.apply(lambda row:polyfit[row.cl2](row.days_from_onset),axis=1)

    # if max val is None, estimate it
    if maxval == None:
        custom_fcn = lambda x, slope, maxval: globals()[fcn_var[varname]](x, slope, maxval)
        parnum = 2
    else:
        custom_fcn = lambda x, slope: globals()[fcn_var[varname]](x, slope, maxval)
        parnum = 1

    
    funfit,funpar = {},{}
    for li,l in list(df.groupby(classvar)):
        l = l.dropna(subset=[varname])
        x = l["days_from_onset"]
        y = l[varname]
        
        coefficients, a = curve_fit(custom_fcn, x, y,p0=[1000]*parnum,method='trf')
                                    # bounds = [[0,3000]]*(int(maxval == None)+1)
        if len(coefficients)>1:
            maxval = coefficients[1]

        # Generate polynomial function based on the coefficients
        funfit.update({li:lambda x: globals()[fcn_var[varname]](x, coefficients[0],  maxval)})
        funpar.update({li:coefficients})

    df["trendline_fun"] = df.apply(lambda row:funfit[row[classvar]](row.days_from_onset),axis=1)
    print(funpar)

    # from matplotlib import pyplot as plt
    # x = np.linspace(0, 1080, 100)
    # y = exp_decay_fcn(x, coefficients[0],  48)
    # plt.plot(x, y)
    # plt.show()
    # cl2
    # 2    77292
    # 1    49320
    df = df.merge(df.groupby('numid')['days_from_onset'].nunique().reset_index(),on='numid',how='left',suffixes=('','_n')).rename(columns={"days_from_onset_n":"dates_unique"})
    df['dates_unique_simple'] = pd.cut(df['dates_unique'],[0,4,8,12,16,20,np.inf],labels=["0-4","5-8","9-12","13-16","17-20","21+"])
    
    if save_flag:
        pdg.to_gbq(df,f"ALS.T{varname}_classes",project_id=project_id,if_exists='replace')
        # df.to_csv(f"data/ALS.T{varname}_classes.csv",index=False)
    
    # build response
    if return_df:
        return {varname:funpar},df
    else:
        return {varname:funpar}
    
# pdg.to_gbq(D[1],"ALS.ToyALSFRS_forpred",project_id=project_id)

class Transform_classes():
    def __init__(self,D=None,from_gbq=True):
        if D is not None:
            self.D = D
        else:
            if from_gbq:
                self.read_gbq()
            else:
                self.read_csv()

    def run(self):
        # set classes and compute thresholds
        self.set_classes()
        self.fetch_threshold_classes()

        # add fields 
        self.D["decline"] = pd.concat([g['class'].map(class_dict[gi]) for gi,g in list(self.D.groupby('variable'))])
        self.D['value'] = self.D['value'].astype(float)
        
        # transform weight to percentage
        self.transform_weight_to_pct()

        # compute quantiles and smooth them 
        self.compute_quantiles()
        self.smooth_quantiles()

        # save
        self.save_csv()

    def transform_weight_to_pct(self,cap=0.05):
        aux = self.D.query('variable == "Weight" and days_from_onset == 0').loc[:,['numid','value']]
        aux = self.D.query('variable == "Weight"').reset_index().merge(aux,on='numid',how='left',suffixes=('','_onset'))
        aux['value'] = np.clip((aux['value'] - aux['value_onset'])/aux['value_onset'],-np.inf,cap)*100
        self.D.loc[aux.loc[:,'index'],'value'] = aux['value'].values

    def compute_quantiles(self):
        def q75(x):
            return x.quantile(0.75)
        def q25(x):
            return x.quantile(0.25)

        self.df2 = self.D.groupby(['days_from_onset','decline','variable']).agg({'value':['median',q25,q75]}).rename(columns={'value':''}).reset_index()
        self.df2.columns = self.df2.columns.map('|'.join).str.strip('|')
    
    def smooth_quantiles(self):
        def smooth(x):
            return x.rolling(window=3,min_periods=1).mean()
        self.df2['median'] = self.df2.groupby(['decline','variable'])['median'].transform(smooth)
        self.df2['q25'] = self.df2.groupby(['decline','variable'])['q25'].transform(smooth)
        self.df2['q75'] = self.df2.groupby(['decline','variable'])['q75'].transform(smooth)
    
    def read_gbq(self):
        D = []
        for var in allvars:
            df = pdg.read_gbq(f"ALS.T{var}_classes",project_id=project_id)
            df['variable'] = var
            D.append(df)
        self.D = pd.concat(D,axis=0,ignore_index=True)
        
    def save_csv(self):
        # self.D.to_csv('app/appdata/T5models_classes.csv',index=False)
        self.df2.to_csv('app/appdata/T5models_classes_iqr.csv',index=False)
        print("saved csv")

    def set_classes(self):
        D = self.D
        # set classes
        D['value'] = D['ALSFRS'].fillna(D['Weight']).fillna(D['Q3']).fillna(D['Bulbar']).fillna(D['Resp100'])
        D['class'] = D['cl2'].fillna(D['cl3'])

        to_drop = ['ALSFRS','Weight','Q3','Bulbar','Resp100','trendline','trendline_juan','cl2','cl3']
        to_drop = [f for f in to_drop if f in D.columns]
        D.drop(columns=to_drop,inplace=True)

        # this is an exception and believe to be an outlier
        # patient 1640 after 990 days 
        self.D = D.query('variable != "Resp100" or numid != 1640 or days_from_onset < 990',inplace=False).reset_index(drop=True)  

    def fetch_threshold_classes(self):
        D = self.D
        # now build the new classes for weight
        d2 = D.query("variable == 'Weight'").reset_index(drop=True)
        D = D.query("variable != 'Weight'").reset_index(drop=True)
        # numid  cl2  days_from_onset value
        d3 = get_threshold_classes(d2,var='value')
        class_dict_inv = {v:i for i,v in class_dict['Weight'].items()}
        d3['class'] = d3['chosen_class'].map(class_dict_inv)
        d3.drop(columns=['chosen_class','weights'],inplace=True)
        self.D = pd.concat([D,d3],axis=0,ignore_index=True)


class TransformWeights():
    def __init__(self):
        # read data
        # self.df = pd.read_csv('data/ALS.TWeight_classes.csv')
        self.df = pdg.read_gbq('ALS.TWeight_classes',project_id=project_id)
        self.threshold = .05

    def run(self):
        self.calculate_deltas()
        self.estimate_delay()
        self.resave_to_gbq()
    
    def calculate_deltas(self):
        dfaux = self.df.query('days_from_onset == 0').loc[:,['numid', 'Weight']].rename(columns={'Weight':'w0'})
        self.df = self.df.merge(dfaux,on=['numid'])
        self.df['DeltaW'] = self.df['Weight'] - self.df['w0']
        self.df['RelDeltaW'] = (self.df['DeltaW'])/self.df['w0']
        print(self.df['RelDeltaW'].describe())

    def estimate_delay(self):
        # estimate the delay
        delay = {}
        for gi,g in list(self.df.sort_values(by=['numid','days_from_onset']).groupby('numid')):
            idx = np.where(g['RelDeltaW']<-self.threshold)[0]
            if len(idx)>0:
                delay.update({gi:[g.iloc[idx[0],2],1]})
            else:
                delay.update({gi:[g['days_from_onset'].max(),0]})

        # convert delay dictionary to dataframe
        delay_df = pd.DataFrame.from_dict(delay, orient='index', columns=['delay', 'threshold_reached']).reset_index().rename(columns={'index':'numid'})

        self.df.drop(columns=['cl2','w0','DeltaW','RelDeltaW'],inplace=True)
        self.df_out = self.df.merge(delay_df,on='numid').rename(columns={'threshold_reached':'cl2'})
        
    def resave_to_gbq(self):
        pdg.to_gbq(self.df_out,'ALS.TWeight_classes',project_id=project_id,if_exists='replace')
