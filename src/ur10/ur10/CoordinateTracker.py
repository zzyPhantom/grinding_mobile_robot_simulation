import cv2
import numpy as np
import mujoco

# 加载MuJoCo环境
m = mujoco.MjModel.from_xml_path('/home/zzy/UR10_sim/src/ur10/ur10/model/universal_robots_ur10e/experment_scene.xml')
d = mujoco.MjData(m)

# 获取相机的参数
camera_id = 1  # 假设你使用的是第2个相机
fovy = m.cam_fovy[camera_id]  # 获取相机的垂直视场角 (Field of View in the Y direction)
image_width = 640  # 图像的宽度
image_height = 480  # 图像的高度

# 计算焦距
focal_length_y = image_height / (2 * np.tan(np.radians(fovy) / 2))
focal_length_x = focal_length_y  # 在大多数情况下，fx = fy

# 计算图像中心点
cx = image_width / 2
cy = image_height / 2

# 构造相机内参矩阵
camera_matrix = np.array([[focal_length_x, 0, cx],
                          [0, focal_length_y, cy],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # 如果没有畸变，可以保持为零

# 加载图像
image = cv2.imread('new_image.jpg')

# 棋盘格尺寸
pattern_size = (4, 4)
square_size = 0.08  # 每个格子的边长，例如25毫米

# 寻找棋盘格角点
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

if ret:
    # 定义棋盘格的3D点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 使用solvePnP计算位姿
    ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if ret:
        # 计算旋转矩阵
        R, _ = cv2.Rodrigues(rvec)

        print("Rotation Matrix:")
        print(R)
        print("Translation Vector:")
        print(tvec)
    else:
        print("位姿估计失败")
else:
    print("未找到棋盘格角点")