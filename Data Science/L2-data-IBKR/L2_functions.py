import csv
import pandas as pd
import numpy as np
import decimal
import os

def create_VP_template_file(symbol, ATL, ATH, step): 
    headers = np.arange( decimal.Decimal(f'{ATL}'), decimal.Decimal(f'{ATH}'), decimal.Decimal(f'{step}')).tolist()
    headers = ['years', 'month', 'day', 'hour'] + headers

    file_exists = os.path.isfile(f'L2/VP_{symbol}.csv')

    if not file_exists :
        with open(f'L2/VP_{symbol}.csv', 'a') as file:
            writer = csv.DictWriter(file,fieldnames=headers)
            writer.writeheader()  # file doesn't exist yet, write a header

        with open(f'L2/VP_{symbol}.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(['','','','']+['0' for k in range(len(headers)-4)])

def extend_VP_headers(symbol,ATL,ATH,step,ATL_old,ATH_old):
    df = pd.read_csv(f'L2/VP_{symbol}.csv')
    
    #add new higher columns
    range_max = int((decimal.Decimal(f'{ATH}')-decimal.Decimal(f'{ATH_old}'))/decimal.Decimal(f'{step}'))
    df[[str(decimal.Decimal(f'{ATH_old}')+decimal.Decimal(f'{step}')*k) for k in range(range_max+1)]]=[[0 for j in range(range_max+1)] for k in range(len(df.index))]
    
    #add new lower columns
    range_min = int((decimal.Decimal(f'{ATL_old}')-decimal.Decimal(f'{ATL}'))/decimal.Decimal(f'{step}'))
    for i in range(1,range_min+1):
        df.insert(4,str(decimal.Decimal(f'{ATL_old}')-decimal.Decimal(f'{step}')*i),[0 for k in range(len(df.index))])
    
    df.to_csv(f'L2/VP_{symbol}.csv', index=False)
    
    df = pd.read_csv('symbols_ATL_ATH_accuracy.csv')
    df.loc[df['symbol']==symbol,'ATL'] = ATL
    df.loc[df['symbol']==symbol,'ATH'] = ATH
    df.to_csv('symbols_ATL_ATH_accuracy.csv', index=False)
    
def add_a_new_row(filename: str):
    df = pd.read_csv(filename, index_col=False)
    ligne = df.iloc[len(df.index)-1].to_list()
    
    with open(filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(ligne)