# parse_data.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file parses the data

import sys
sys.path.append('app')
import pandas as pd 
import pandas_gbq as pdg
from utils.constants import *
from utils.utils import *
import numpy as np
from utils.parser_utils import parse_classes_data,TransformWeights,Transform_classes

# new datasets
paths = ["data/FinalclasesAndCoxModels/tals_nodup_classes.csv",
         "data/FinalclasesAndCoxModels/q3_nodup_classes.csv",
         "data/FinalclasesAndCoxModels/bulb_nodup_classes.csv",
         "data/FinalclasesAndCoxModels/Resp_nodup_classes.csv",
         "data/WeightVel_classes.csv"
    ]

# training data only
D = [pd.read_csv(p) for p in paths]

# parse the data
funpar = parse_classes_data(D,0,'ALSFRS',48,True,False,True,classvar='cl3',return_df=False)
funpar.update(parse_classes_data(D,1,'Q3',4,False,False,True,return_df=False))
funpar.update(parse_classes_data(D,2,'Bulbar',12,False,False,True,classvar='cl3',return_df=False))
funpar.update(parse_classes_data(D,3,'Resp100',None,False,False,True,classvar='cl3',return_df=False))
funpar.update(parse_classes_data(D,4,'Weight',None,False,False,True,return_df=False))

# enrich weight classes
TransformWeights().run()

# transform classes into smoothed IQR
Transform_classes().run() # this will automatically save the file

# load the survival curves
dict_surv = pd.read_excel("data/survProb_TALSFRS.xlsx",sheet_name=None)
# df_surv = pd.concat(dict_surv.values(),ignore_index=True)
df_surv = pd.DataFrame()
for i,df in dict_surv.items():
    df["class"] = i
    df_surv = pd.concat([df_surv,df],ignore_index=True)

# export avg only
# df_surv_simple = df_surv.loc[:,["class","Ps","days"]]
pdg.to_gbq(df_surv.astype(str),"ALS.survProb_TALSFRS_simple",project_id=project_id,if_exists='replace')


df_surv = df_surv.merge(df_surv.groupby("class")["Ps"].count().reset_index().rename(columns={"Ps":"n"}),
              on="class",
              how="left")

# generate data for the survival curves
# ciu = psU-Ps / cil = psL-Ps
# ci = 1.96*se
# std = se*sqrt(n)

df_surv["stdU"] = (df_surv["PsU"] - df_surv["Ps"])/1.96*np.sqrt(df_surv["n"])
df_surv["stdL"] = -(df_surv["PsL"] - df_surv["Ps"])/1.96*np.sqrt(df_surv["n"])
df_surv["std"] = (df_surv["PsU"] - df_surv["PsL"])/1.96*np.sqrt(df_surv["n"])

ns = 100
df_surv["survival"] = df_surv.apply(lambda row:np.random.normal(row["Ps"],row["std"],ns),axis=1)
df_surv = df_surv.explode("survival").reset_index(drop=True)

to_drop = ["PsU","PsL","n","stdU","stdL","std","Ps"]
df_surv.drop(columns=to_drop,inplace=True,axis=1)

df_surv["survival_clipped"] = df_surv["survival"].clip(0,1)

pdg.to_gbq(df_surv.astype(str),"ALS.survProb_TALSFRS",project_id=project_id,if_exists='replace')
