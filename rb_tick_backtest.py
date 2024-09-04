import numpy as np
import pandas as pd
from datetime import time
from multiprocessing import Pool,cpu_count,Value,Manager
import multiprocessing
import os
import talib as ta
from tqdm import tqdm
import json
from datetime import datetime,timedelta
from multiprocessing import Pool,cpu_count
import multiprocessing

BASE_DIR = os.path.dirname(__file__)
FILE_NAME = os.path.basename(__file__).split('.')[0]
RES_PATH = BASE_DIR+f'/result/{FILE_NAME}/'
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)

END_0 = time(15,0)
END_1 = time(23,0)


def backtest(args,is_fitting = True):

    data = args[0]
    result = args[1]
    l_benifit = args[2]
    s_benifit = args[3]

    pos = 0
    long_diff = []
    short_diff = []
    total_diff = []
    direction = []
    changed = []
    pos_price = 0
    min_price = 0 
    max_price = 0

    detail = {}
    detail['date'] = []
    detail['slope'] = []
    detail['grad'] = []
    detail['atr'] = []
    detail['min_diff'] = []
    detail['max_diff'] = []
    detail['pos_price'] = []
    detail['direction'] = []
    detail['offset'] = []
    detail['diff'] = []


    for row in tqdm(data.iterrows()):
        changed_sign = False
        sign = row[1]['sign']   
        
        # if row[0].time() != END_0 and row[0].time() != END_1:
        tick_date =  datetime.fromtimestamp(row[1]['datatime'].timestamp()) - timedelta(hours=8)
        tick_time = tick_date.time()
        if tick_time != time(20,59) and tick_time != time(8,59):
            if pos == 0:
                if row[1]['min_diff']>=4:
                    sign *= -1
                    changed_sign = True
                elif len(total_diff)>=3:
                    if total_diff[-1] <0 and total_diff[-2]<0 and total_diff[-3]<0:
                        sign *= -1
                        changed_sign = True
                    elif sum(total_diff[-2:]) < -32:
                        sign *= -1
                        changed_sign = True
        
                if sign == 1:
                    pos = 1
                    pos_price = row[1]['ask_price']
                    max_price = pos_price
                    min_price = pos_price
                    if not is_fitting:
                        detail['date'].append(row[1]['date'])
                        detail['pos_price'].append(pos_price)
                        detail['slope'].append(row[1]['slope'])
                        detail['grad'].append(row[1]['grad'])
                        detail['atr'].append(row[1]['atr'])
                        detail['max_diff'].append(row[1]['max_diff'])
                        detail['min_diff'].append(row[1]['min_diff'])
                        detail['direction'].append(1)
                        detail['offset'].append('open')
                        detail['diff'].append(0)
                    if changed_sign:
                        print(row[1]['date'])
                elif sign == -1:
                    pos = -1
                    pos_price = row[1]['bid_price']
                    min_price = pos_price
                    max_price = pos_price
                    if not is_fitting:
                        detail['date'].append(row[1]['date'])
                        detail['pos_price'].append(pos_price)
                        detail['slope'].append(row[1]['slope'])
                        detail['grad'].append(row[1]['grad'])
                        detail['atr'].append(row[1]['atr'])
                        detail['max_diff'].append(row[1]['max_diff'])
                        detail['min_diff'].append(row[1]['min_diff'])
                        detail['direction'].append(-1)
                        detail['offset'].append('open')
                        detail['diff'].append(0)
                    if changed_sign:
                        print(row[1]['date'])
                continue

            elif pos == 1:
                # 用high和low计算收盘
                max_price = max(max_price, row[1]['last_price'])
                min_price = min(min_price,row[1]['last_price'])
                #close_drop 为下单后的历史高价与当前收盘价的差值
                close_drop = max_price - row[1]['bid_price']

                diff_0 = row[1]['bid_price'] - pos_price
                
                if close_drop >= l_benifit:
                    long_diff.append(diff_0)
                    pos = 0
                    total_diff.append(diff_0)
                    direction.append(1)
                    changed.append(diff_0)

                    if not is_fitting:
                        detail['date'].append(row[1]['date'])
                        detail['pos_price'].append(row[1]['bid_price'])
                        detail['slope'].append(row[1]['slope'])
                        detail['grad'].append(row[1]['grad'])
                        detail['atr'].append(row[1]['atr'])
                        detail['max_diff'].append(row[1]['max_diff'])
                        detail['min_diff'].append(row[1]['min_diff'])
                        detail['direction'].append(-1)
                        detail['offset'].append('close')
                        detail['diff'].append(diff_0)
                continue

            elif pos == -1:
                # 用high和low计算收盘
                max_price = max(max_price, row[1]['last_price'])
                min_price = min(min_price,row[1]['last_price'])
                close_drop = row[1]['ask_price'] - min_price
            
                diff_0 =  pos_price - row[1]['ask_price']

                if close_drop >= s_benifit:
                    short_diff.append(diff_0)
                    pos = 0
                    total_diff.append(diff_0)
                    direction.append(-1)
                    changed.append(-1*diff_0)

                    if not is_fitting:
                        detail['date'].append(row[1]['date'])
                        detail['pos_price'].append(row[1]['ask_price'])
                        detail['slope'].append(row[1]['slope'])
                        detail['grad'].append(row[1]['grad'])
                        detail['atr'].append(row[1]['atr'])
                        detail['max_diff'].append(row[1]['max_diff'])
                        detail['min_diff'].append(row[1]['min_diff'])
                        detail['direction'].append(1)
                        detail['offset'].append('close')
                        detail['diff'].append(diff_0)
                continue

    if is_fitting == True:
        res = {"long_diff":sum(long_diff),"long_trades":len(long_diff),
                "short_diff":sum(short_diff),"short_trades":len(short_diff)}
        result.append({f"{l_benifit}_{s_benifit}":res})
    else:
        base_name = os.path.basename(__file__).split('.')[0]
        file_name = RES_PATH + base_name +f'-backtest_result.csv'
        res_dict = {'direction':direction,'total_diff':total_diff,'changed':changed}
        res_pd = pd.DataFrame(res_dict)
        res_pd.to_csv(file_name,index=None)

        detail_name = RES_PATH + base_name +f'-backtest_detail.csv'
        detail_pd = pd.DataFrame(detail)
        detail_pd.to_csv(detail_name,index=None)

        print("long_diff",sum(long_diff),"long_trades",len(long_diff))
        print("short_diff",sum(short_diff),"short_trades",len(short_diff))


def write_fitting_result(result):
    base_name = os.path.basename(__file__).split('.')[0]
    now = datetime.now().strftime('%Y-%m-%d %H-%M')
    file_name = RES_PATH+base_name +f'_{now}.jsonl'
    best_param_id = 0
    best_benifit = 0
    for i,res in enumerate(result):
        for k,v in res.items():
            if v['long_diff']+v['short_diff'] > best_benifit:
                best_benifit = v['long_diff']+v['short_diff']
                best_param_id = i
    write_best_param(result[best_param_id])

    with open(file_name,'w') as f:
        for data in result:
            # 将每个数据对象转换为JSON字符串并写入文件
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def write_best_param(result):
    base_name = os.path.basename(__file__).split('.')[0]
    file_name = file_name = RES_PATH+base_name +'-best_param.jsonl'
    if os.path.exists(file_name):
        with open(file_name,'r') as f:
            base_param = json.load(f)
        base_diff = list(base_param.values())[0]
        new_diff = list(result.values())[0]
        base_benifit = base_diff['long_diff']+base_diff['short_diff']
        new_benifit = new_diff['long_diff']+new_diff['short_diff']
        if base_benifit > new_benifit:
            best_param = base_param
        else:
            best_param = result
    else:
        best_param = result
    
    with open(file_name,'w') as f:
        json.dump(best_param,f)

def read_best_parm():
    base_name = os.path.basename(__file__).split('.')[0]
    file_name = file_name = RES_PATH+base_name +'-best_param.jsonl'
    if os.path.exists(file_name):
        with open(file_name,'r') as f:
            base_param = json.load(f)
            config_str = list(base_param.keys())[0]
            configs = [int(i) for i in config_str.split('_')]
        return configs
    else:
        raise FileExistsError

def multi_backtest(input_path,data,worker_num =2,**config):
    manager = Manager()
    result = manager.list()
    
    config_list = set_param(**config)
    param =[(data,result,i[0],i[1]) for i in config_list]
    pool = Pool(worker_num)
    pool.map(backtest, param)
    pool.close()
    pool.join()
    write_fitting_result(result)

def set_param(**configs):
    print(configs)
    l_benifit = configs['l_benifit']
    s_benifit = configs['s_benifit']

    l_benifit_list = list(range(l_benifit[0],l_benifit[1],l_benifit[2]))
    s_benifit_list = list(range(s_benifit[0],s_benifit[1],s_benifit[2]))
    config_list = []
    for i in l_benifit_list:
        for j in s_benifit_list:
            config_list.append([i,j])
    return config_list


def argmax(x):
    return x.argmax()
def argmin(x):
    return x.argmin()
def get_diff(x):
    if abs(x[0])>abs(x[1]):
        return x[0]
    else:
        return x[1]
#下单逻辑在这里改
def get_data(path,start=20000,end=50000,period=1):
    data = pd.read_csv(path)
    data['datatime'] = pd.to_datetime(data['datatime'])
    data['max_diff'] = data['last_price'] - data['min']
    data['min_diff'] = data['last_price'] - data['max']

    h1 = data[(data['last_price']<=data['bid_price'])&(data['bid_volume']>4*data['ask_volume'])&(data['slope']>0)].index
    l1 =  data[(data['ask_volume']>5*data['bid_volume'])&(data['slope']<0)&(data['grad']>0)].index
    data['sign']=0
    data.loc[h1,'sign'] = 1
    data.loc[l1,'sign'] = -1

    return data[start:end]


if __name__ == "__main__":
    """
        参数顺序data,result,benifit,loss
    """
    is_fitting = False
    input_path = "D:/Code/jupyter_project/data_analysis/bar_analysis/bar_backtest/rb数据分析/rb2405_tick_new.csv"
    start = 0
    end = 1000000
    data = get_data(input_path,start=start,end=end,period=10)
    config = {
        "l_benifit":[16,19,1],
        "s_benifit":[16,19,1]
    }

    if is_fitting:
        multi_backtest(input_path,data,worker_num=4,**config)
    else:
        param = read_best_parm()
        result = {}
        args = (data,result,param[0],param[1])
        backtest(args=args,is_fitting=is_fitting )
