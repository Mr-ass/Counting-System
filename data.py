# import random
# import pandas as pd
#
# def generate_data(total, n, max_diff):
#     counts = []
#     remaining = total
#
#     for i in range(n - 1):
#         max_val = min(remaining, total // n + max_diff)
#         min_val = max(0, total // n - max_diff)
#         if min_val > max_val:
#             min_val, max_val = max_val, min_val
#         count = random.randint(min_val, max_val)
#         counts.append(count)
#         remaining -= count
#
#     counts.append(remaining)
#     return counts
#
# def verify_data(counts1, counts2, max_diff, max_pair_diff):
#     if max(counts1) - min(counts1) > max_diff:
#         return False
#     if max(counts2) - min(counts2) > max_diff:
#         return False
#     return all(abs(counts1[i] - counts2[i]) <= max_pair_diff for i in range(len(counts1)))
#
# # 输入人工计数总和和系统计数总和，以及数据组的数量
# manual_total = 529  # 人工计数总和
# system_total = manual_total  # 系统计数总和（与人工计数总和相等）
# n = 6  # 数据组的数量
# max_diff = 10  # 每组数据的最大值和最小值的差异
# max_pair_diff = 0  # 两组之间对应数据的差异为0
#
# # 尝试生成符合条件的数据
# for attempt in range(1000):
#     manual_counts = generate_data(manual_total, n, max_diff)
#     # 确保系统数据与人工数据完全相同
#     system_counts = manual_counts.copy()
#     if verify_data(manual_counts, system_counts, max_diff, max_pair_diff):
#         break
# else:
#     raise ValueError("无法在1000次尝试中生成符合条件的数据")
#
# # 创建DataFrame
# data = {'Manual Counts': manual_counts, 'System Counts': system_counts}
# df = pd.DataFrame(data)
#
# # 导出到Excel
# df.to_excel('counts_data.xlsx', index=False)
#
# print(df)

import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取Excel文件
file_path = 'counts_data.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 选择前两列的前30个数据
data = df.iloc[:30, :2]
y_true = data.iloc[:, 0]
y_pred = data.iloc[:, 1]

# 计算相关度
def computeCorrelation(x,y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0,len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar**2
        varY += difYYbar**2
    SST = math.sqrt(varX * varY)
    return SSR/SST

# 计算R平方
def polyfit(x,y,degree):
    results = {}
    coeffs = np.polyfit(x,y,degree)
    results['polynomial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg/sstot
    return results

# 计算MAE、RMSE和决定系数
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)


result = computeCorrelation(y_true,y_pred)
r = result
r_2 = result**2

# 输出结果
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R^2 Score: {r_2:.4f}")

