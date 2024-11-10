# 特征提取2

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(42)
data = pd.read_csv('out\code_data\sz.300760_15m_data.csv')

# 定义特征和预测目标
features = ['open', 'high', 'low', 'close', 'volume', 'amount']
X = data[features]

# 定义两个预测目标：open_diff 和 close_diff
target_open_diff = data['open_diff']
target_close_diff = data['close_diff']

# 数据划分为训练集和测试集
X_train, X_test, y_train_open_diff, y_test_open_diff = train_test_split(X, target_open_diff, test_size=0.2, random_state=42)
_, _, y_train_close_diff, y_test_close_diff = train_test_split(X, target_close_diff, test_size=0.2, random_state=42)

# 创建和训练随机森林模型来预测 open_diff
rf_model_open_diff = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_open_diff.fit(X_train, y_train_open_diff)

# 创建和训练随机森林模型来预测 close_diff
rf_model_close_diff = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_close_diff.fit(X_train, y_train_close_diff)

# 提取特征重要性
importance_open_diff = rf_model_open_diff.feature_importances_
importance_close_diff = rf_model_close_diff.feature_importances_

# 将特征重要性存储到 DataFrame 中
importance_df_open_diff = pd.DataFrame({
    'Feature': features,
    'Importance_open_diff': importance_open_diff
}).sort_values(by='Importance_open_diff', ascending=False)

importance_df_close_diff = pd.DataFrame({
    'Feature': features,
    'Importance_close_diff': importance_close_diff
}).sort_values(by='Importance_close_diff', ascending=False)

# 绘制特征重要性对比图
plt.figure(figsize=(12, 8))

# 收盘收益率 (open_diff) 特征重要性
plt.subplot(1, 2, 1)
plt.barh(importance_df_open_diff['Feature'], importance_df_open_diff['Importance_open_diff'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance for open_diff Prediction')
plt.gca().invert_yaxis()  # 反转 y 轴，使重要性高的特征排在顶部

# 开盘收益率 (close_diff) 特征重要性
plt.subplot(1, 2, 2)
plt.barh(importance_df_close_diff['Feature'], importance_df_close_diff['Importance_close_diff'], color='salmon')
plt.xlabel('Importance')
plt.title('Feature Importance for Open open_diff Prediction')
plt.gca().invert_yaxis()  # 反转 y 轴，使重要性高的特征排在顶部

plt.tight_layout()
plt.show()

# 输出特征重要性数据
print("Feature Importance for open_diff Prediction:")
print(importance_df_open_diff)

print("\nFeature Importance for close_diff Prediction:")
print(importance_df_close_diff)
