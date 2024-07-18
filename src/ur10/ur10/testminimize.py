# import numpy as np
# from scipy.optimize import minimize

# def objective(x):
#     p1, p2 = x
#     return p1**2 + p2**2

# def constraint1(x):
#     p1, p2 = x
#     return ((r * (p1 + p2)) / 2) - q1

# def constraint2(x):
#     p1, p2 = x
#     return ((r * (p1 - p2)) / l) - q2

# # 给定的参数和目标值
# q1 = 1.0  # 根据具体情况设置
# q2 = 1.5  # 根据具体情况设置
# r = 0.01   # 根据具体情况设置
# l = 0.5   # 根据具体情况设置

# # 初始猜测值
# x0 = np.array([0.0, 0.0])

# # 设置约束
# constraints = ({'type': 'eq', 'fun': constraint1},
#                {'type': 'eq', 'fun': constraint2})

# # 最小化问题
# result = minimize(objective, x0, constraints=constraints)

# # 输出最优解
# p1_opt = result.x[0]
# p2_opt = result.x[1]

# print("最小化 p1^2 + p2^2 的结果：")
# print(f"p1 = {p1_opt}")
# print(f"p2 = {p2_opt}")
# print(q2 -((p1_opt+p2_opt)*r)/l)
# print(q1 -((p1_opt+p2_opt)*r)/2)

import numpy as np
from scipy.optimize import minimize

# 给定的参数和目标值
q1 = 1.0  # 根据具体情况设置
q2 = 1.5  # 根据具体情况设置
r = 0.01   # 根据具体情况设置
l = 0.5   # 根据具体情况设置

# 初始猜测值
x0 = np.array([0.0, 0.0])

cons = ({'type': 'eq', 'fun': lambda x: (r * (x[0] + x[1])) / 2 - q1},
        {'type': 'eq', 'fun': lambda x: (r * (x[0] - x[1])) / l - q2})

obj = lambda x: x[0]**2 + x[1]**2
# 最小化问题
result = minimize(obj, x0, constraints=cons)

# 输出最优解
p1_opt = result.x[0]
p2_opt = result.x[1]

print("最小化 p1^2 + p2^2 的结果：")
print(f"p1 = {p1_opt}")
print(f"p2 = {p2_opt}")
print(q2 -((p1_opt-p2_opt)*r)/l)
print(q1 -((p1_opt+p2_opt)*r)/2)