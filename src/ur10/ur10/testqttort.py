import numpy as np

# 四元数到旋转矩阵的转换
def quat_to_rot_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

# 第一个四元数
quat1 = [0.707105, 0, 0, -0.707105]
R1 = quat_to_rot_matrix(quat1)

# 打印第一个旋转矩阵
print("Rotation matrix from first quaternion:")
print(R1)

# 第二个四元数
quat2 = [0, 0, 1, 0]
R2 = quat_to_rot_matrix(quat2)

# 打印第二个旋转矩阵
print("Rotation matrix from second quaternion:")
print(R2)

# 组合两个旋转矩阵
R_combined = R1 @ R2

# 打印组合后的旋转矩阵
print("Combined rotation matrix:")
print(R_combined)