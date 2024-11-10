# 加权平均

import pandas as pd
import os

if __name__ == "__main__":
    # 读取股票代码
    si2 = pd.read_csv('out/codes/si_codes.csv')['code'].tolist()
    st2 = pd.read_csv('out/codes/st_codes.csv')['code'].tolist()
    si2 = ['sh.000001']
    st2 = ['sh.600000','sh.600028']  # 只使用一个股票代码进行测试

    # 存储所有股票的数据
    all_data = []

    for si in si2:
        # 读取对应股票代码的预测结果文件
        a = pd.read_csv(f'out/pre/{si}_d_predicted_results.csv')

        # 计算加权平均
        open = a['Predicted Open Price'] * 1
        close = a['Predicted Close Price'] *1

        # 将结果合并为一个列表
        stock_data = [si]  # 股票代码
        for i in range(len(a)):
            stock_data.append(open[i]) 
            stock_data.append(close[i]) 

        # 将单只股票的数据添加到 all_data 列表
        all_data.append(stock_data)

    for st in st2:
        # 读取对应股票代码的预测结果文件
        a = pd.read_csv(f'out/pre/{st}_15m_predicted_results.csv')
        b = pd.read_csv(f'out/pre/{st}_d_predicted_results.csv')

        # 计算加权平均
        open = a['Predicted Open Price'] * 0.7 + b['Predicted Open Price'] * 0.3
        close = a['Predicted Close Price'] * 0.7 + b['Predicted Close Price'] * 0.3

        # 将结果合并为一个列表
        stock_data = [st]  # 股票代码
        for i in range(len(a)):
            stock_data.append(open[i]) 
            stock_data.append(close[i]) 

        # 将单只股票的数据添加到 all_data 列表
        all_data.append(stock_data)

    # 列名
    columns = ['Stock Code']
    for i in range(1, len(a) + 1):  # 假设a和b的行数相同
        columns.extend([f'Open{i}', f'Close{i}'])

    # 创建 DataFrame
    result_df = pd.DataFrame(all_data, columns=columns)

    # 保存到 CSV 文件
    result_df.to_csv('out/pre/all_stocks_with_prices.csv', index=False)
