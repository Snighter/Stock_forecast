# 平稳性检验

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

# 平稳性检验：ADF 测试
def adf_test(series):
    """执行ADF平稳性检验"""
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] < 0.05  # p-value < 0.05 表示平稳
    }

# 平稳性检验：KPSS 测试
def kpss_test(series):
    """执行KPSS平稳性检验"""
    result = kpss(series, regression='c')
    return {
        'KPSS Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[3],
        'Stationary': result[1] >= 0.05  # p-value >= 0.05 表示平稳
    }

if __name__ == "__main__":
    x_d = ['open', 'close', 'pctChg', 'open_pctChg']
    x_15m = ['open_diff','close_diff']
    
    si2 = pd.read_csv('out/codes/si_codes.csv')['code'].tolist()
    st2 = pd.read_csv('out/codes/st_codes.csv')['code'].tolist()

    results_d = []  # 存储结果的列表

    for code in si2 + st2:
        file_path = f'out/code_data/{code}_d_data.csv'
        try:
            data = pd.read_csv(file_path)
            # 存储每个变量的收益率
            returns_dict = {var: data[var].astype(float) for var in x_d if var in data.columns}

            for var, returns in returns_dict.items():
                print(f"\n--- 处理 {code} 的 {var} 数据 ---")
                adf_results = adf_test(returns)
                kpss_results = kpss_test(returns)

                is_stationary = adf_results['Stationary'] and kpss_results['Stationary']

                results_d.append({
                    'Code': code,
                    'Variable': var,
                    'ADF Statistic': adf_results['ADF Statistic'],
                    'ADF p-value': adf_results['p-value'],
                    'KPSS Statistic': kpss_results['KPSS Statistic'],
                    'KPSS p-value': kpss_results['p-value'],
                    'Stationary': is_stationary,
                    'Type': '平稳' if is_stationary else '非平稳'
                })

        except FileNotFoundError:
            print(f"文件 {file_path} 不存在。")

    results_df1 = pd.DataFrame(results_d)
    results_df1.to_csv('out/stationarity_results_d.csv', index=False)

    results_15m = []

    for code in st2:
        file_path = f'out/code_data/{code}_15m_data.csv'
        try:
            data = pd.read_csv(file_path)
            # 存储每个变量的收益率
            returns_dict = {var: data[var].astype(float) for var in x_15m if var in data.columns}

            for var, returns in returns_dict.items():
                print(f"\n--- 处理 {code} 的 {var} 数据 ---")
                adf_results = adf_test(returns)
                kpss_results = kpss_test(returns)

                is_stationary = adf_results['Stationary'] and kpss_results['Stationary']

                results_15m.append({
                    'Code': code,
                    'Variable': var,
                    'ADF Statistic': adf_results['ADF Statistic'],
                    'ADF p-value': adf_results['p-value'],
                    'KPSS Statistic': kpss_results['KPSS Statistic'],
                    'KPSS p-value': kpss_results['p-value'],
                    'Stationary': is_stationary,
                    'Type': '平稳' if is_stationary else '非平稳'
                })

        except FileNotFoundError:
            print(f"文件 {file_path} 不存在。")

    results_df2 = pd.DataFrame(results_15m)
    results_df2.to_csv('out/stationarity_results_15m.csv', index=False)

    sum_cnt_d = len(si2) + len(st2)

    for xd in x_d:
        count = sum(1 for result in results_d if result['Stationary'] and result['Variable'] == xd)
        print(f"\n总共有 {count}/{sum_cnt_d} 只股票日线{xd}平稳")

    sum_cnt_15m = len(st2)

    for x15m in x_15m:
        count = sum(1 for result in results_15m if result['Stationary'] and result['Variable'] == x15m)
        print(f"\n总共有 {count}/{sum_cnt_15m} 只股票15min线{x15m}平稳")

