import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import talib as ta
import numpy as np
from datetime import datetime,timedelta,time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

origion = pd.read_csv('RB2405.csv')
origion['datetime'] = pd.to_datetime(origion['datetime'])
# origion.set_index('datetime',inplace=True)
origion = origion.iloc[:,[1,3,4,8,9]]
origion.head()

for i,v in origion.iterrows():
    t0 = datetime.fromtimestamp(origion.iloc[i,0].timestamp()) - timedelta(hours = 8)
    if t0.time() == time(23,0) or t0.time() == time(15,0):
        t1 = datetime.fromtimestamp(origion.iloc[i+1,0].timestamp()) - timedelta(hours = 8) - timedelta(minutes =1)
        origion.iloc[i,0] = t1

tick_data = pd.read_csv("D:/Code/jupyter_project/data_analysis/tick_analysis/tick_backtest_2/data/rb2405_tick.csv")
tick_data['date'] = pd.to_datetime(tick_data['date'])
tick_data.head()

def get_new_date(x):
    x = datetime.fromtimestamp(x.timestamp()) - timedelta(hours = 8)
    x = x.replace(second=0)
    return x

tick_data['datatime'] = tick_data['date'].apply(get_new_date)

tick_data['max'] = np.nan
tick_data['min'] = np.nan
tick_data.head()

start = datetime(2024,3,15)
end = datetime(2024,4,26)
origion = origion[(origion['datetime']>=start)&(origion['datetime']<=end)]
origion.set_index('datetime',inplace=True)
origion.head()


data = origion.resample('1min').agg(
    {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last'
    }
).dropna()
data.head()

data['max'] = data['high'].rolling(100).max()
data['min'] = data['low'].rolling(100).min()

for i,v in tqdm(data.iterrows()):
    tick_ids = tick_data[tick_data['datatime'] == i].index
    tick_data.loc[tick_ids,'max'] = v[4]
    tick_data.loc[tick_ids,'min'] = v[5]

