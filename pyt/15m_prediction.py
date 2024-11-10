# 15m 线预测

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# 归一化数据
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    return data_scaled, scaler

# 使用数据训练模型
def train_model(X_open_diff, y_open_diff, X_close_diff, y_close_diff, n_steps):
    # 模型结构
    model_open = Sequential()
    model_open.add(GRU(64, return_sequences=True, input_shape=(n_steps, 1)))
    model_open.add(Dropout(0.2))
    model_open.add(GRU(64, return_sequences=True))
    model_open.add(Dropout(0.2))
    model_open.add(GRU(32, return_sequences=False))
    model_open.add(Dropout(0.2))
    model_open.add(Dense(1))
    
    model_close = Sequential()
    model_close.add(GRU(64, return_sequences=True, input_shape=(n_steps, 1)))
    model_close.add(Dropout(0.2))
    model_close.add(GRU(64, return_sequences=True))
    model_close.add(Dropout(0.2))
    model_close.add(GRU(32, return_sequences=False))
    model_close.add(Dropout(0.2))
    model_close.add(Dense(1))
    
    # 编译模型
    optimizer = Adam(learning_rate=0.0005)
    model_open.compile(optimizer=optimizer, loss='mean_squared_error')
    model_close.compile(optimizer=optimizer, loss='mean_squared_error')

    # 训练模型
    model_open.fit(X_open_diff, y_open_diff, epochs=50, batch_size=32, verbose=0)
    model_close.fit(X_close_diff, y_close_diff, epochs=50, batch_size=32, verbose=0)

    return model_open, model_close

# 还原为开盘价和收盘价
def recover_prices(last_open, last_close, open_diff, close_diff):
    next_open = last_open + open_diff
    next_close = last_close + close_diff
    return next_open, next_close

# 计算 RMSE
def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return rmse

# 读取目录中所有CSV文件
def load_all_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('15m_data.csv')]
    data_files = []
    for file in files:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path)
        data_files.append(data)
    return data_files

# 主程序
if __name__ == "__main__":
    input_directory = 'out/code_data'  # 数据目录
    output_directory = 'out/pre'  # 预测结果输出目录

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 读取所有数据文件
    data_files = load_all_files(input_directory)

    for data in data_files:
        # 提取特征数据 {'open', 'close', 'volume', 'amount'}
        open_data = data['open']
        close_data = data['close']
        volume_data = data['volume']
        amount_data = data['amount']
        
        # 计算开盘和收盘差值（假设已经计算好 `open_diff` 和 `close_diff`）
        open_diff = data['open_diff']
        close_diff = data['close_diff']

        # 归一化数据
        open_diff_scaled, scaler_open_diff = normalize_data(open_diff)
        close_diff_scaled, scaler_close_diff = normalize_data(close_diff)

        # 使用前20个数据点训练模型
        n_steps = 5 * 4 * 4  # 使用前5天数据点
        X_open_diff, y_open_diff = prepare_data(open_diff_scaled, n_steps)
        X_close_diff, y_close_diff = prepare_data(close_diff_scaled, n_steps)

        # 划分训练集和测试集
        split = int(len(X_open_diff) * 0.8)
        X_train_open_diff, X_test_open_diff = X_open_diff[:split], X_open_diff[split:]
        y_train_open_diff, y_test_open_diff = y_open_diff[:split], y_open_diff[split:]

        X_train_close_diff, X_test_close_diff = X_close_diff[:split], X_close_diff[split:]
        y_train_close_diff, y_test_close_diff = y_close_diff[:split], y_close_diff[split:]

        # 调整输入数据形状为 [样本数, 时间步长, 特征数]
        X_train_open_diff = X_train_open_diff.reshape((X_train_open_diff.shape[0], X_train_open_diff.shape[1], 1))
        X_test_open_diff = X_test_open_diff.reshape((X_test_open_diff.shape[0], X_test_open_diff.shape[1], 1))

        X_train_close_diff = X_train_close_diff.reshape((X_train_close_diff.shape[0], X_train_close_diff.shape[1], 1))
        X_test_close_diff = X_test_close_diff.reshape((X_test_close_diff.shape[0], X_test_close_diff.shape[1], 1))

        # 训练模型
        model_open, model_close = train_model(X_train_open_diff, y_train_open_diff, X_train_close_diff, y_train_close_diff, n_steps)

        # 进行未来几天的预测
        future_days = 5  # 预测未来5个时间步
        predicted_open_diffs = []
        predicted_close_diffs = []
        predicted_open_prices = []
        predicted_close_prices = []

        # 获取最后的开盘价和收盘价
        last_open_price = data['open'].iloc[-1]
        last_close_price = data['close'].iloc[-1]

        # 使用最后的 n_steps 天数据作为初始输入
        current_input_open_diff = open_diff_scaled[-n_steps:]
        current_input_close_diff = close_diff_scaled[-n_steps:]

        for _ in range(future_days):
            current_input_open_diff = current_input_open_diff.reshape((1, n_steps, 1))
            predicted_open_diff = model_open.predict(current_input_open_diff)
            
            current_input_close_diff = current_input_close_diff.reshape((1, n_steps, 1))
            predicted_close_diff = model_close.predict(current_input_close_diff)

            # 反归一化
            predicted_open_diff = scaler_open_diff.inverse_transform(predicted_open_diff)
            predicted_close_diff = scaler_close_diff.inverse_transform(predicted_close_diff)

            # 还原为开盘价和收盘价
            next_open, next_close = recover_prices(last_open_price, last_close_price, predicted_open_diff[0, 0], predicted_close_diff[0, 0])

            # 存储结果
            predicted_open_diffs.append(predicted_open_diff[0, 0])
            predicted_close_diffs.append(predicted_close_diff[0, 0])
            predicted_open_prices.append(next_open)
            predicted_close_prices.append(next_close)

            # 更新输入数据
            current_input_open_diff = np.append(current_input_open_diff[0, 1:], predicted_open_diff[0, 0]).reshape(n_steps, 1)
            current_input_close_diff = np.append(current_input_close_diff[0, 1:], predicted_close_diff[0, 0]).reshape(n_steps, 1)

            # 更新收盘价和开盘价
            last_open_price = next_open
            last_close_price = next_close

        # 输出预测结果到CSV
        output_df = pd.DataFrame({
            'Predicted Open Diff': predicted_open_diffs,
            'Predicted Close Diff': predicted_close_diffs,
            'Predicted Open Price': predicted_open_prices,
            'Predicted Close Price': predicted_close_prices
        })

        # 保存结果到CSV文件
        output_file = os.path.join(output_directory, f"{data['code'].iloc[0]}_15m_predicted_results.csv")
        output_df.to_csv(output_file, index=False)

        print(f"预测结果已保存至 {output_file}")

        # 计算 RMSE
        y_pred_open = model_open.predict(X_test_open_diff)  # 开盘价格差预测值
        y_pred_close = model_close.predict(X_test_close_diff)  # 收盘价格差预测值

        # 反归一化
        y_pred_open = scaler_open_diff.inverse_transform(y_pred_open)
        y_pred_close = scaler_close_diff.inverse_transform(y_pred_close)

        # 计算 RMSE
        rmse_open = calculate_rmse(y_test_open_diff, y_pred_open)

        print(rmse_open)
