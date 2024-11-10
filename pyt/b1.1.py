# 数据收集

import baostock as bs
import os
import shutil
import pandas as pd

def fetch_k_15m(code, start_date, end_date):
    rs = bs.query_history_k_data_plus(
        code,
        "date,time,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="15"
    )

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 计算 open_diff 和 close_diff
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    
    df['open_diff'] = df['open'].diff().fillna(0).round(2)  # 与上一次的 open 差值
    df['close_diff'] = df['close'].diff().fillna(0).round(2)  # 与上一次的 close 差值
    
    return df

def fetch_k_d(code, start_date, end_date):
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start_date,
        end_date=end_date,
        frequency="d"
    )
    
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 计算 open 收益率
    df['open'] = df['open'].astype(float)
    df['open_pctChg'] = df['open'].pct_change().fillna(0) * 100  # open 收益率
    
    return df

def clear_or_create_directory(directory):
    # 检查文件夹是否存在
    if os.path.exists(directory):
        # 如果存在，清空文件夹
        shutil.rmtree(directory)
        
    # 创建一个新的文件夹
    os.makedirs(directory)

def main():
    # 登录系统
    lg = bs.login()
        
    # 定义查询的日期范围
    start_date = '2024-01-01'
    end_date = '2024-11-30'

    clear_or_create_directory('out')

    os.makedirs('out/code_data')
        
    # 遍历每个股票代码
    for code in si + st:
        result = fetch_k_d(code, start_date, end_date)
        
        # 将结果输出到CSV文件
        result.to_csv(f"out/code_data/{code}_d_data.csv", index=False)

    for code in st:
        start_date = '2024-10-01'

        result = fetch_k_15m(code, start_date, end_date)
        
        # 将结果输出到CSV文件
        result.to_csv(f"out/code_data/{code}_15m_data.csv", index=False)
    
    bs.logout()

    os.makedirs('out/codes')

    si_df = pd.DataFrame(si, columns=['code'])
    si_df.to_csv('out/codes/si_codes.csv', index=False)

    st_df = pd.DataFrame(st, columns=['code'])
    st_df.to_csv('out/codes/st_codes.csv', index=False)

if __name__ == "__main__":
    # 指数
    si = ["sh.000001","sz.399001","sz.399006"] 

    # 股票
    st = pd.read_excel('in/d1.xls', dtype=str)
    st = st['code'].tolist()

    # 为股票代码添加前缀
    st = [
        f'sh.{code}' if code.startswith('6') else 
        f'sz.{code}' for code in st
    ]
    
    main()