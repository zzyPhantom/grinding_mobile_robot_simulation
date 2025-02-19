import time
import numpy as np
from scipy.spatial.transform import Rotation

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/home/zzy/UR10_sim/src/ur10/ur10/model/universal_robots_ur10e/scene.xml')
d = mujoco.MjData(m)

global p_targer, link_lengths, err_k

p_targer = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 初始目标位姿
link_lengths = [0.1807, 0.6127, 0.5716, 0.1742, 0.1199, 0.1166] # UR10e链长
err_k = 100 # 阻尼最小二乘法误差系数

def get_target_position():
    # 更新目标位姿
    global p_targer
    # x = float(input("Enter X position: "))
    # y = float(input("Enter Y position: "))
    # z = float(input("Enter Z position: "))
    p_targer += 0.01

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

def Txd(s):
    T = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    return T

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

def FK(q, L, Tbase, Ttool):
    # Forward Kinematics calculation
    H = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Tz(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]), 
            np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Ry(q[3]), np.dot(Ty(L[3]), 
            np.dot(Rz(q[4]), np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))

    return H

def Jacobian(q, L, Tbase, Ttool):
    # Calculate forward kinematics
    H = FK(q, L, Tbase, Ttool)
    R = H[:3, :3]  # Rotation matrix part

    # 1st column of Jacobian
    J1p = np.dot(Tbase, np.dot(Rzd(q[0]), np.dot(Ty(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Rz(q[3]), np.dot(Ty(L[3]), np.dot(Ry(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J1r = np.dot(J1p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J1 = Jcol(J1r)

    # 2nd column of Jacobian
    J2p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Ty(L[0]), np.dot(Ryd(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Rz(q[3]), np.dot(Ty(L[3]), np.dot(Ry(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J2r = np.dot(J2p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J2 = Jcol(J2r)

    # 3rd column of Jacobian
    J3p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Ty(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ryd(q[2]), np.dot(Tz(L[2]), np.dot(Rz(q[3]), np.dot(Ty(L[3]), np.dot(Ry(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J3r = np.dot(J3p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J3 = Jcol(J3r)

    # 4th column of Jacobian
    J4p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Ty(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Rzd(q[3]), np.dot(Ty(L[3]), np.dot(Ry(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J4r = np.dot(J4p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J4 = Jcol(J4r)

    # 5th column of Jacobian
    J5p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Ty(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Rz(q[3]), np.dot(Ty(L[3]), np.dot(Ryd(q[4]),
           np.dot(Tz(L[4]), np.dot(Ry(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J5r = np.dot(J5p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J5 = Jcol(J5r)

    # 6th column of Jacobian
    J6p = np.dot(Tbase, np.dot(Rz(q[0]), np.dot(Ty(L[0]), np.dot(Ry(q[1]), np.dot(Tz(L[1]),
           np.dot(Ry(q[2]), np.dot(Tz(L[2]), np.dot(Rz(q[3]), np.dot(Ty(L[3]), np.dot(Ry(q[4]),
           np.dot(Tz(L[4]), np.dot(Ryd(q[5]), np.dot(Ty(L[5]), Ttool)))))))))))))
    J6r = np.dot(J6p, np.vstack((np.hstack((np.linalg.inv(R), np.zeros((3, 1)))), np.array([[0, 0, 0, 1]]))))
    J6 = Jcol(J6r)

    # Combine columns to form the full Jacobian
    J = np.concatenate((J1, J2, J3, J4, J5, J6), axis=1)

    return J

# 关节逆运动学计算函数
def compute_inverse_kinematics(L, p_goal, k):
    # 使用 mujoco 的 ik_solver
    q_current = d.qpos[9:15]
    q_updated = q_current.copy()  # 保证不修改原始输入的值
    
    p_curr = FK(q_updated, L, np.eye(4), np.eye(4))
    p_curr = np.concatenate((p_curr[:3, 3], [0, 0, 0]))  # 获取当前末端位置
    
    err = p_goal - p_curr  # 计算误差
    
    u = 0.2
    I_m = np.ones((6, 6))
    
    while np.linalg.norm(err[:3]) > 0.01:  # 当位置误差大于 0.01 时继续迭代
        jacob = Jacobian(q_updated, L, np.eye(4), np.eye(4))  # 计算雅可比矩阵
        del_err = err / k  # 计算调节后的误差
        
        J_DLS = np.dot(jacob.T, np.linalg.pinv(np.dot(jacob, jacob.T) + (u ** 2) * I_m))  # 计算阻尼最小二乘法的伪逆
        
        del_q = np.dot(J_DLS, del_err)  # 计算关节角度变化量
        q_updated += del_q  # 更新关节角度
        
        p_curr = FK(q_updated, L, np.eye(4), np.eye(4))  # 计算更新后的末端位置
        p_curr = np.concatenate((p_curr[:3, 3], [0, 0, 0]))  # 获取当前末端位置
        err = p_goal - p_curr  # 计算更新后的位置误差
    
    return q_updated

if __name__ == "__main__":
  with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 300:
      step_start = time.time()

      # get_target_position()

      # 设置初始关节角度
      # desired_joint_angles[:6] -= 0.0001  # 示例：设置每个关节的角度
      # d.ctrl[:] = desired_joint_angles
      # d.ctrl[:6] = compute_inverse_kinematics(link_lengths, p_targer, err_k)
      
      if (time.time() - step_start) % 1 < 0.01:
        xyz = d.qpos[0:3]
        quat = d.qpos[3:7]
        Tbase = homogeneous_matrix(xyz, quat)
        p_curr = FK(d.qpos[9:15], link_lengths, Tbase, np.eye(4))
        p_curr = homogeneous_matrix_to_array(p_curr)
        # print(p_curr)
        print(d.qpos)
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