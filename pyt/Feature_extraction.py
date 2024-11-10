# 特征提取

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(42)
data = pd.read_csv('out\code_data\sh.000001_d_data.csv')

# 定义特征和预测目标
features = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']
X = data[features]

# 定义两个预测目标：pctchg 和 open_pctchg
target_pctChg = data['pctChg']  # 预测目标 1：收盘收益率
target_open_pctChg = data['open_pctChg']  # 预测目标 2：开盘收益率

# 数据划分为训练集和测试集
X_train, X_test, y_train_pctChg, y_test_pctChg = train_test_split(X, target_pctChg, test_size=0.2, random_state=42)
_, _, y_train_open_pctChg, y_test_open_pctChg = train_test_split(X, target_open_pctChg, test_size=0.2, random_state=42)

# 创建和训练随机森林模型来预测 pctChg
rf_model_pctChg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_pctChg.fit(X_train, y_train_pctChg)

# 创建和训练随机森林模型来预测 open_pctChg
rf_model_open_pctChg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_open_pctChg.fit(X_train, y_train_open_pctChg)

# 提取特征重要性
importance_pctChg = rf_model_pctChg.feature_importances_
importance_open_pctChg = rf_model_open_pctChg.feature_importances_

# 将特征重要性存储到 DataFrame 中
importance_df_pctChg = pd.DataFrame({
    'Feature': features,
    'Importance_PctChg': importance_pctChg
}).sort_values(by='Importance_PctChg', ascending=False)

importance_df_open_pctChg = pd.DataFrame({
    'Feature': features,
    'Importance_Open_PctChg': importance_open_pctChg
}).sort_values(by='Importance_Open_PctChg', ascending=False)

# 绘制特征重要性对比图
plt.figure(figsize=(12, 8))

# 收盘收益率 (pctChg) 特征重要性
plt.subplot(1, 2, 1)
plt.barh(importance_df_pctChg['Feature'], importance_df_pctChg['Importance_PctChg'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance for PctChg Prediction')
plt.gca().invert_yaxis()  # 反转 y 轴，使重要性高的特征排在顶部

# 开盘收益率 (open_pctChg) 特征重要性
plt.subplot(1, 2, 2)
plt.barh(importance_df_open_pctChg['Feature'], importance_df_open_pctChg['Importance_Open_PctChg'], color='salmon')
plt.xlabel('Importance')
plt.title('Feature Importance for Open PctChg Prediction')
plt.gca().invert_yaxis()  # 反转 y 轴，使重要性高的特征排在顶部

plt.tight_layout()
plt.show()

# 输出特征重要性数据
print("Feature Importance for PctChg Prediction:")
print(importance_df_pctChg)

print("\nFeature Importance for Open PctChg Prediction:")
print(importance_df_open_pctChg)
