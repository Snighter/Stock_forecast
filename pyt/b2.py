# 数据预处理:检查数据是否有缺失值和类型不匹配现象(未输出即为无异常)

import pandas as pd

def check_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("有缺失值")
        print(missing_values[missing_values > 0])

def check_data_types(df, expected_types):
    for column, expected_type in expected_types.items():
        actual_type = df[column].dtype
        if actual_type != expected_type:
            print(f"列 '{column}' 的类型不匹配：预期 {expected_type}, 实际 {actual_type}")
    

def preprocess_si_data(file_path):
    si_data = pd.read_csv(file_path)
    
    # 检查缺失值
    check_missing_values(si_data)
    
    # 检查数据类型
    expected_types = {
        'date': 'object',
        'code': 'object',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'preclose': 'float64',
        'volume': 'int64',
        'amount': 'float64',
        'pctChg': 'float64',
        'open_pctChg': 'float64'
    }
    check_data_types(si_data, expected_types)

def preprocess_st_data(file_path):
    st_data = pd.read_csv(file_path)
    
    # 检查缺失值
    check_missing_values(st_data)
    
    # 检查数据类型
    expected_types = {
        'date': 'object',
        'time': 'int64',
        'code': 'object',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'int64',
        'amount': 'float64',
        'open_diff': 'float64',
        'close_diff': 'float64'
    }
    check_data_types(st_data, expected_types)

if __name__ == "__main__":
    si2 = pd.read_csv('out/codes/si_codes.csv')['code'].tolist()
    st2 = pd.read_csv('out/codes/st_codes.csv')['code'].tolist()

    for code in si2 + st2:
        preprocess_si_data(f'out/code_data/{code}_d_data.csv')  # 指数列表读取数据

    for code in st2:
        preprocess_st_data(f'out/code_data/{code}_15m_data.csv')  # 个股列表读取数据
