import sys, pandas as pd
R=sys.argv[1]; sys.path.insert(0,R)
from model import run_exp, default_params
import utils as utl
from brian2 import ms
neu_sugar=[720575940624963786,720575940630233916,720575940637568838,720575940638202345,720575940617000768,720575940630797113,720575940632889389,720575940621754367,720575940621502051,720575940640649691,720575940639332736,720575940616885538,720575940639198653,720575940620900446,720575940617937543,720575940632425919,720575940633143833,720575940612670570,720575940628853239,720575940629176663,720575940611875570]
p=dict(default_params); p['n_run']=4
out=sys.argv[2]
run_exp('sugarR',neu_sugar,out,R+'/2023_03_23_completeness_630_final.csv',R+'/2023_03_23_connectivity_630_final.parquet',params=p,n_proc=4,force_overwrite=True)
df=utl.load_exps([out+'/sugarR.parquet'])
print('spikes',len(df))
r,_=utl.get_rate(df,t_run=p['t_run'],n_run=p['n_run'])
r=r.sort_values('sugarR',ascending=False)
print('active neurons',(r['sugarR']>0).sum())
MN9=720575940660219265
print('MN9 rate', r['sugarR'].get(MN9,'n/a'))
print(r.head(15))
