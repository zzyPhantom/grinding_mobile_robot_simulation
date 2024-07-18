import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
import cv2
import threading

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/home/zzy/UR10_sim/src/ur10/ur10/model/universal_robots_ur10e/experment_scene.xml')
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 640)

global p_target, link_lengths, err_k, min_pos_err, dT_last, step_time, q_last, force_history

link_lengths = [0.1807, 0.6127, 0.5716, 0.1742, 0.1199, 0.1166] # UR10e链长
car_params = [0.1, 0.1, 1] # 小车参数（左轮半径，右轮半径，轴距）
eef_params = [1.5] # 末端工具参数（重量）
err_k = 5# 阻尼最小二乘法误差系数
min_pos_err = 0.01 # 迭代位置误差阈值
p_target = np.zeros(6)
dT_last = np.zeros(6)
q_last = np.zeros(8)
step_time = 0.0
cam_interval = 1/30.0  # 相机显示间隔时间，30FPS
display_interval = 5  # figure显示间隔时间,秒

FORCE_HISTORY_SIZE = 1

#定义机器人模型
E1 = rtb.ET.tx(-0.4)
E2 = rtb.ET.tx()
E3 = rtb.ET.Rz()
E4 = rtb.ET.tx(0.4)
E5 = rtb.ET.Rz()
E6 = rtb.ET.tz(link_lengths[0])
E7 = rtb.ET.Ry()
E8 = rtb.ET.tz(link_lengths[1])
E9 = rtb.ET.Ry()
E10 = rtb.ET.tz(link_lengths[2])
E11 = rtb.ET.Ry()
E12 = rtb.ET.ty(link_lengths[3])
E13 = rtb.ET.Rz()
E14 = rtb.ET.tz(link_lengths[4])
E15 = rtb.ET.Ry()
E16 = rtb.ET.ty(link_lengths[5])
ur10m = E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15 * E16

def transform_to_xyzrpy(T):
    # 提取平移部分
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    
    # 提取旋转矩阵部分
    R = T[:3, :3]
    
    # 计算欧拉角
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([x, y, z, roll, pitch, yaw])

force_history = []
def get_target_position(force):
    # 更新目标位姿
    global p_target
    global force_history

    # force = d.sensor('eef_force').data.copy()

    # 如果力历史记录还没满，直接添加新数据
    if len(force_history) < FORCE_HISTORY_SIZE:
        force_history.append(force)
    else:
    # 如果力历史记录已满，删除最早的力数据，然后添加新数据
        force_history.pop(0)
        force_history.append(force)

    # 计算平均力
    avg_force = np.mean(force_history, axis=0)

    p_target[0] = d.joint('workpiece_joint').qpos[0].copy() - 0.36 + 0.000*avg_force[2]
    p_target[1] = d.joint('workpiece_joint').qpos[1].copy() - 0.3
    p_target[2] = d.joint('workpiece_joint').qpos[2].copy() + 0.2
    # p_target[0] = 1.1 
    # p_target[1] = -0.3
    # p_target[2] = 0.7

    p_target[3:] = [0, 0.2, -0.7]

def homogeneous_matrix_to_array(H):
    """Convert a homogeneous matrix to an array"""
    # Extract position (translation part)
    position = H[:3, 3]

    # Extract rotation (rotation matrix part)
    rotation_matrix = H[:3, :3]

    # Convert rotation matrix to Euler angles (XYZ convention)
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz')

    # Combine position and euler angles into a single array
    p_rpy_array = np.concatenate((position, euler_angles))

    return p_rpy_array

def Rz(q):
    """生成绕 Z 轴旋转的变换矩阵"""
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array([
        [cq, -sq, 0, 0],
        [sq, cq, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def Ry(q):
    """生成绕 Y 轴旋转的变换矩阵"""
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array([
        [cq, 0, sq, 0],
        [0, 1, 0, 0],
        [-sq, 0, cq, 0],
        [0, 0, 0, 1]
    ])

def Rx(q):
    """生成绕 X 轴旋转的变换矩阵"""
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array([
        [1, 0, 0, 0],
        [0, cq, -sq, 0],
        [0, sq, cq, 0],
        [0, 0, 0, 1]
    ])

def Tz(d):
    """生成沿 Z 轴平移的变换矩阵"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])

def Ty(d):
    """生成沿 Y 轴平移的变换矩阵"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, d],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def Tx(d):
    """生成沿 X 轴平移的变换矩阵"""
    return np.array([
        [1, 0, 0, d],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def Ryd(q):
    Sq = np.sin(q)
    Cq = np.cos(q)
    
    T = np.array([
        [-Sq, 0, Cq, 0],
        [0, 0, 0, 0],
        [-Cq, 0, -Sq, 0],
        [0, 0, 0, 0]
    ])
    
    return T

def Rydd(q):
    Sq = np.sin(q)
    Cq = np.cos(q)
    
    T = np.array([
        [-Cq, 0, -Sq, 0],
        [0, 0, 0, 0],
        [Sq, 0, -Cq, 0],
        [0, 0, 0, 0]
    ])
    
    return T

def Rzd(q):
    Sq = np.sin(q)
    Cq = np.cos(q)
    
    T = np.array([
        [-Sq, -Cq, 0, 0],
        [Cq, -Sq, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    return T

def Rzdd(q):
    Sq = np.sin(q)
    Cq = np.cos(q)
    
    T = np.array([
        [-Cq, Sq, 0, 0],
        [-Sq, -Cq, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    return T

def Txd(s):
    T = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    return T

def Hwl(dql):
    Sq = np.sin(dql*car_params[0])
    Cq = np.cos(dql*car_params[0])
    Sql = np.sin(-dql*car_params[0]/car_params[2])
    Cql = np.cos(-dql*car_params[0]/car_params[2])

    T = np.array([
        [Cql, -Sql, 0, Cq/2],
        [Sql, Cql, 0, Sq/2],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    return T

def Hwr(dqr):
    Sq = np.sin(dqr*car_params[1])
    Cq = np.cos(dqr*car_params[1])
    Sqr = np.sin(dqr*car_params[1]/car_params[2])
    Cqr = np.cos(dqr*car_params[1]/car_params[2])

    T = np.array([
        [Cqr, -Sqr, 0, Cq/2],
        [Sqr, Cqr, 0, Sq/2],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    return T

def vec(matrix):

    return matrix.flatten(order='F')

def Jcol(T):
    # Extract the Jacobian column from the transformation matrix T
    J = np.array([T[0, 3], T[1, 3], T[2, 3], T[2, 1], T[0, 2], T[1, 0]]).reshape((-1,1))
    return J

def quaternion_to_rotation_matrix(q):

    q_w, q_x, q_y, q_z = q
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def homogeneous_matrix(xyz, quaternion):

    x, y, z = xyz
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def force_sensor_data(H_eef, ddT):
    
    data = d.sensor('eef_force').data.copy()

    # 定义重力加速度
    g = 9.81  # m/s^2

    # 计算工具重力在基座标系下的力
    tool_gravity_base = np.array([0, 0, eef_params[0] * g])
    #先不计算加速度产生的力（此方法需要验证）
    # tool_extForce_base = tool_gravity_base - np.array([ddT[0], ddT[1], ddT[2]]) * eef_params[0]

    # 将工具重力转换到末端坐标系
    R_eef = H_eef[:3, :3]  # 从齐次变换矩阵中提取旋转矩阵
    tool_gravity_eef = np.zeros(3)
    tool_gravity_eef = np.dot(tool_gravity_base, np.dot(R_eef, np.dot(Rz(-np.pi/2)[:3, :3], Ry(np.pi)[:3, :3])))

    # 计算工具重心产生的力矩
    # tool_gravity_eef[3:] = np.cross(tool_com, tool_gravity_eef[:3])

    # 减去工具重力的影响
    corrected_data = data - tool_gravity_eef

    # print ('m',tool_gravity_eef)
    # print('r',d.sensor('eef_tool_acc').data)

    #转回基座标系
    corrected_data_base = np.dot(R_eef.T, corrected_data)

    return corrected_data_base

def FK(q, L, Tbase, Ttool):
    # Forward Kinematics calculation
    H = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]), 
            np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), 
            np.dot(Rz(q[4]), np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))

    return H

def Jacobian(q, L, Tbase, Ttool):
    # Calculate forward kinematics
    H = FK(q[:6], L, Tbase, Ttool)
    R = H[:3, :3]  # Rotation matrix part

    # 1st column of Jacobian
    J1p = np.dot(Tbase, np.dot(Rzd(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), np.dot(Rz(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J1r = np.dot(J1p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J1 = Jcol(J1r)

    # 2nd column of Jacobian
    J2p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ryd(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), np.dot(Rz(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J2r = np.dot(J2p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J2 = Jcol(J2r)

    # 3rd column of Jacobian
    J3p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ryd(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), np.dot(Rz(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J3r = np.dot(J3p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J3 = Jcol(J3r)

    # 4th column of Jacobian
    J4p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ryd(q[3]), np.dot(Ty(L[3]), np.dot(Rz(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J4r = np.dot(J4p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J4 = Jcol(J4r)

    # 5th column of Jacobian
    J5p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), np.dot(Rzd(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J5r = np.dot(J5p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J5 = Jcol(J5r)

    # 6th column of Jacobian
    J6p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), np.dot(Rz(q[4]),
           np.dot(Tz(L[4]), np.dot(Ryd(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J6r = np.dot(J6p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J6 = Jcol(J6r)

    # Combine columns to form the full Jacobian
    J = np.concatenate((J1, J2, J3, J4, J5, J6), axis=1)

    return J

def Manipulability_Jacobian(q, jacob):
    # Manipulability Jacobian
    q = [q[0], q[1], q[2], q[3], q[4], q[5]]
    He = ur10m.hessiane(q)
    He = np.array([np.concatenate((layer[:, 2:], layer[:, :2]), axis=1) for layer in He])
    # 定义 J, H1, H2, ..., Hn 和 m
    J = jacob

    # 计算 JJ^T 和它的逆矩阵
    JJ_T = np.dot(J, J.T)
    JJ_T_inv = np.linalg.inv(JJ_T)
    vec_JJ_T_inv = vec(JJ_T_inv)

    # 计算行列式
    det_JJ_T = np.linalg.det(JJ_T)

    # 计算行列式的平方根
    m = np.sqrt(det_JJ_T)

    # 计算每一项并存储在列表中
    MJ = []
    for H in He:
        JH_T = np.dot(J, H.T)
        vec_JH_T = vec(JH_T).T
        result = m * np.dot(vec_JH_T, vec_JJ_T_inv)
        MJ.append(result)

    # 将结果转换为数组或所需的形状
    MJ = np.array(MJ)

    return MJ

# 关节逆运动学计算函数
def compute_inverse_kinematics(L, p_goal, k):
    global dT_last, q_last, step_time
    k_positioncon = 0.9
    k_manipulability = 0.3
    Pi = 0.1   # influence distance in which to activate the damper
    Ps = 0.01  # stopping distance

    # 使用 mujoco 的 ik_solver
    q_current = d.qpos[9:15].copy() # 保证不修改原始输入的值
    q_updated = [q_current[0], q_current[1], q_current[2], q_current[3], q_current[4], q_current[5]]  

    Tbase = np.eye(4)
    p_curr_H = FK(q_updated[:6], L, Tbase, np.eye(4))
    p_curr = homogeneous_matrix_to_array(p_curr_H)  # 获取当前末端位置
    
    err = p_goal - p_curr  # 计算误差
    
    u = 0.2
    I_m = np.ones((6, 6))

    jacob = Jacobian(q_updated, L, Tbase, np.eye(4))  # 计算雅可比矩阵
    Mjacob = Manipulability_Jacobian(q_updated, jacob)  # 计算可操纵性雅各比矩阵

    # 计算阻尼最小二乘法的伪逆
    # del_err = err / k  # 计算调节后的误差
    # J_DLS = np.dot(jacob.T, np.linalg.pinv(np.dot(jacob, jacob.T) + (u ** 2) * I_m))  
    # del_q = np.dot(J_DLS, del_err)  # 计算关节角度变化量

    # Quadratic Programmming
    # 初始猜测值
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    I = np.eye(len(q0))
    err_vel = err * 100  #末端的期望速度

    # 计算关节限制
    P = np.zeros(6)
    q = q_updated[1] + np.pi/2
    distance_to_low = q
    distance_to_high = np.pi - q
    if abs(distance_to_low) < abs(distance_to_high):
        if distance_to_low < 0:
            P[1] = 0
        else:
            P[1] = distance_to_low
    else:
        if distance_to_high < 0:
            P[1] = np.pi
        else:
            P[1] = distance_to_high

    # 最小化问题
    if q_updated[1] < 0:
        cons = ({'type': 'eq', 'fun': lambda dq: np.dot(jacob, dq).flatten() - err_vel}, 
                {'type': 'ineq', 'fun': lambda dq: ((np.pi/2 + q_updated[1]) - Ps)/(Pi - Ps) * k_positioncon - abs(dq[1])})
    else:
        cons = ({'type': 'eq', 'fun': lambda dq: np.dot(jacob, dq).flatten() - err_vel},
                {'type': 'ineq', 'fun': lambda dq: ((np.pi/2 - q_updated[1]) - Ps)/(Pi - Ps) * k_positioncon + abs(dq[1])})
    obj = lambda dq: (np.dot(np.dot(dq.T, I), dq) * k_manipulability / 2) + np.dot(Mjacob.T, dq) 
    q_vel = minimize(obj, q0, constraints=cons)

    # 计算关节角度变化量
    del_q = q_vel.x * (d.time - step_time)
    step_time = d.time
    
    q_updated[:6] += del_q[:6]  # 更新关节角度

    ddT = 0.0

    force_data = force_sensor_data(p_curr_H, ddT)

    return q_updated, force_data, p_curr

if __name__ == "__main__":
    #显示传感器数据
    plt.ion()  # 开启交互模式

    # Initialize empty lists to store force data components
    x_data, y_data, z_data, position_data = [], [], [], []
    time_points = []

    # 初始化绘图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        #初始化数据
        force_data = np.zeros(3)
        last_display_time = d.time
        last_cam_time = d.time

        start = time.time()
        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()

            if d.time <= 2:
                d.ctrl[:6] = [0, 0.377, 1.16, 0.1, 0.1, 0.1]  # 末端执行器横向
                # d.ctrl[:6] = [0, 0.5, 1.0, -1.63, -1.45, 0.1]   # 末端执行器纵向
                d.ctrl[-1] = 100

            if d.time > 2:
                get_target_position(force_data)
                IK, force_data, eef_pos = compute_inverse_kinematics(link_lengths, p_target, err_k)
                d.ctrl[:6] = IK

                current_cam_time = d.time
                elapsed_cam_time = current_cam_time - last_cam_time
                if elapsed_cam_time >= cam_interval:
                    last_cam_time = current_cam_time
                    # 更新相机图像
                    r.update_scene(d, camera="base_cam")
                    cam_img = r.render()
                    cv2.imshow('Image from base camera', cam_img)
                    cv2.waitKey(1)  # 确保图像显示

                # 如果力历史记录还没满，直接添加新数据
                if len(time_points) < display_interval * 500:
                    x_data.append(force_data[0])
                    y_data.append(force_data[1])
                    z_data.append(force_data[2])
                    position_data.append(eef_pos[0])
                    time_points.append(d.time)
                else:
                # 如果力历史记录已满，删除最早的力数据，然后添加新数据
                    x_data.pop(0)
                    y_data.pop(0)
                    z_data.pop(0)
                    position_data.pop(0)
                    time_points.pop(0)
                    x_data.append(force_data[0])
                    y_data.append(force_data[1])
                    z_data.append(force_data[2])
                    position_data.append(eef_pos[0])
                    time_points.append(d.time)

                current_time = d.time
                elapsed_time = current_time - last_display_time
                # 检查是否达到显示间隔
                if elapsed_time >= display_interval:
                    # 子图像绘制力数据
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()
                    ax4.clear()
                    plt.pause(0.001)

                    ax1.plot(time_points, x_data, label='X Force', color='b')
                    ax2.plot(time_points, y_data, label='Y Force', color='g')
                    ax3.plot(time_points, z_data, label='Z Force', color='r')
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel('Force')
                    ax1.set_title('Force Sensor Data')
                    ax1.legend()
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Force')
                    ax2.set_title('Force Sensor Data')
                    ax2.legend()
                    ax3.set_xlabel('Time')
                    ax3.set_ylabel('Force')
                    ax3.set_title('Force Sensor Data')
                    ax3.legend()

                    # 绘制位移数据
                    ax4.plot(time_points, position_data, label='X position', color='c')
                    ax4.set_xlabel('Time')
                    ax4.set_ylabel('Position')
                    ax4.set_title('Position Data')
                    ax4.legend()

                    plt.pause(0.001)
                    last_display_time = current_time

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)