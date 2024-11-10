import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math

# 数据准备与处理
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# 反还原收益率为收盘价
def recover_close_price(last_close, pctchg):
    return last_close * (1 + pctchg / 100)

# 计算 RMSE
def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return rmse

# 读取目录中所有后缀为 `d_data.csv` 的文件
def load_all_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('d_data.csv')]
    data_files = []
    for file in files:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path)
        data_files.append(data)
    return data_files

# 主程序
if __name__ == "__main__":
    # 读取所有数据文件
    directory = 'out/code_data'  # 假设存放数据的目录
    data_files = load_all_files(directory)

    output_directory = 'out/pre'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 遍历每个文件进行处理
    for data in data_files:
        # 读取收盘价和开盘价收益率
        pctchg = data['pctChg'].astype(float).values
        open_pctchg = data['open_pctChg'].astype(float).values  # 计算开盘价收益率
        last_open_price = data['open'].iloc[-1]  # 获取最后一天的开盘价
        last_close_price = data['close'].iloc[-1]  # 获取最后一天的收盘价

        # 归一化数据
        scaler_pctchg = MinMaxScaler(feature_range=(0, 1))
        pctchg_scaled = scaler_pctchg.fit_transform(pctchg.reshape(-1, 1))

        scaler_open_pctchg = MinMaxScaler(feature_range=(0, 1))
        open_pctchg_scaled = scaler_open_pctchg.fit_transform(open_pctchg.reshape(-1, 1))

        # 准备训练数据
        n_steps = 20  # 使用前20天的收益率预测下一天
        X_pctchg, y_pctchg = prepare_data(pctchg_scaled, n_steps)
        X_open_pctchg, y_open_pctchg = prepare_data(open_pctchg_scaled, n_steps)

        # 划分训练集和测试集
        split = int(len(X_pctchg) * 0.8)
        X_train_pctchg, X_test_pctchg = X_pctchg[:split], X_pctchg[split:]
        y_train_pctchg, y_test_pctchg = y_pctchg[:split], y_pctchg[split:]
        
        X_train_open_pctchg, X_test_open_pctchg = X_open_pctchg[:split], X_open_pctchg[split:]
        y_train_open_pctchg, y_test_open_pctchg = y_open_pctchg[:split], y_open_pctchg[split:]

        # 调整输入数据形状为 [样本数, 时间步长, 特征数]
        X_train_pctchg = X_train_pctchg.reshape((X_train_pctchg.shape[0], X_train_pctchg.shape[1], 1))
        X_test_pctchg = X_test_pctchg.reshape((X_test_pctchg.shape[0], X_test_pctchg.shape[1], 1))

        X_train_open_pctchg = X_train_open_pctchg.reshape((X_train_open_pctchg.shape[0], X_train_open_pctchg.shape[1], 1))
        X_test_open_pctchg = X_test_open_pctchg.reshape((X_test_open_pctchg.shape[0], X_test_open_pctchg.shape[1], 1))

        # 构建 GRU 模型
        model_pctchg = Sequential()
        model_pctchg.add(GRU(64, return_sequences=True, input_shape=(n_steps, 1)))
        model_pctchg.add(Dropout(0.2))
        model_pctchg.add(GRU(64, return_sequences=True))
        model_pctchg.add(Dropout(0.2))
        model_pctchg.add(GRU(32, return_sequences=False))
        model_pctchg.add(Dropout(0.2))
        model_pctchg.add(Dense(1))

        model_open_pctchg = Sequential()
        model_open_pctchg.add(GRU(64, return_sequences=True, input_shape=(n_steps, 1)))
        model_open_pctchg.add(Dropout(0.2))
        model_open_pctchg.add(GRU(64, return_sequences=True))
        model_open_pctchg.add(Dropout(0.2))
        model_open_pctchg.add(GRU(32, return_sequences=False))
        model_open_pctchg.add(Dropout(0.2))
        model_open_pctchg.add(Dense(1))

        # 编译模型
        optimizer = Adam(learning_rate=0.0005)
        model_pctchg.compile(optimizer=optimizer, loss='mean_squared_error')
        model_open_pctchg.compile(optimizer=optimizer, loss='mean_squared_error')

        # 训练模型
        model_pctchg.fit(X_train_pctchg, y_train_pctchg, epochs=100, batch_size=32, verbose=0)
        model_open_pctchg.fit(X_train_open_pctchg, y_train_open_pctchg, epochs=100, batch_size=32, verbose=0)

        # 进行未来几天的预测
        future_days = 5  # 预测未来5天
        predicted_pctchgs = []  # 存储预测的收益率
        predicted_open_pctchgs = []  # 存储预测的开盘收益率
        predicted_open_prices = []
        predicted_close_prices = []

        # 使用最后的 n_steps 天数据作为初始输入
        current_input_pctchg = pctchg_scaled[-n_steps:]
        current_input_open_pctchg = open_pctchg_scaled[-n_steps:]

        for _ in range(future_days):
            current_input_pctchg = current_input_pctchg.reshape((1, n_steps, 1))
            predicted_pctchg = model_pctchg.predict(current_input_pctchg)

            current_input_open_pctchg = current_input_open_pctchg.reshape((1, n_steps, 1))
            predicted_open_pctchg = model_open_pctchg.predict(current_input_open_pctchg)

            # 反归一化
            predicted_pctchg = scaler_pctchg.inverse_transform(predicted_pctchg)
            predicted_open_pctchg = scaler_open_pctchg.inverse_transform(predicted_open_pctchg)

            # 还原为收盘价
            last_open_price = recover_close_price(last_open_price, predicted_open_pctchg[0, 0])
            last_close_price = recover_close_price(last_close_price, predicted_pctchg[0, 0])
            predicted_pctchgs.append(predicted_pctchg[0, 0])
            predicted_open_pctchgs.append(predicted_open_pctchg[0, 0])
            predicted_open_prices.append(last_open_price)
            predicted_close_prices.append(last_close_price)
            
            # 更新输入数据，用于下一次预测
            current_input_pctchg = np.append(current_input_pctchg[0, 1:], predicted_pctchg[0, 0]).reshape(n_steps, 1)
            current_input_open_pctchg = np.append(current_input_open_pctchg[0, 1:], predicted_open_pctchg[0, 0]).reshape(n_steps, 1)

        # 生成未来日期
        last_date = pd.to_datetime(data['date'].iloc[-1])
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

        # 输出预测结果到CSV
        output_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted PctChg': predicted_pctchgs,
            'Predicted Open PctChg': predicted_open_pctchgs,
            'Predicted Open Price': predicted_open_prices,
            'Predicted Close Price': predicted_close_prices
        })

        # 保存结果到CSV文件
        output_file = os.path.join(output_directory, f"{data['code'].iloc[0]}_d_predicted_results.csv")
        output_df.to_csv(output_file, index=False)

        # 计算 RMSE 对开盘和收盘价格的预测结果
        y_pred_close = model_pctchg.predict(X_test_pctchg)  # 收盘价格的预测值
        y_pred_open = model_open_pctchg.predict(X_test_open_pctchg)  # 开盘价格的预测值

        # 反归一化
        y_pred_close = scaler_pctchg.inverse_transform(y_pred_close)
        y_pred_open = scaler_open_pctchg.inverse_transform(y_pred_open)

        # 计算 RMSE
        rmse_close = calculate_rmse(y_test_pctchg, y_pred_close)
        rmse_open = calculate_rmse(y_test_open_pctchg, y_pred_open)

        print(f"RMSE for predicted close price: {rmse_close}")
        print(f"RMSE for predicted open price: {rmse_open}")

        print(f"预测结果已保存至 {output_file}")
