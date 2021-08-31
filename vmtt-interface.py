#/usr/bin/python3
import numpy as np

def cam2Inertial(P_b_i_i, P_c_b_b, R_b_c, R_b_i, epsilon):
    P_c_b_i = R_b_i@P_c_b_b
    P_c_i_i = P_c_b_i + P_b_i_i
    R_c_b = R_b_c.T
    R_c_i = R_b_i@R_c_b
    z_c = P_c_i_i[2] / (R_c_i@epsilon)[2]
    # print(z_c)
    P_t_c_i = -1*z_c * R_c_i@epsilon
    # print(P_t_c_i)
    P_t_i_i = P_t_c_i + P_c_i_i
    return P_t_i_i

def getRotMatrix(roll,pitch,yaw):
    phi = deg2rad(roll)
    theta = deg2rad(pitch)
    psi = deg2rad(yaw)
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi),0],
                    [0,0,1]])
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0,1,0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    R_x = np.array([[1,0,0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])
    return R_z@R_y@R_x

def deg2rad(deg):
    return deg * np.pi/180

if __name__ == '__main__':
    R_b_c = np.array([[0,-1,0],
                      [1,0,0],
                      [0,0,1]])
    R_b_i = getRotMatrix(0,0,0)
    P_b_i_i = np.array([0,0,-1]).T
    P_c_b_b = np.array([.166, 0, .068]).T
    epsilon = np.array([0,0,1]).T
    print(cam2Inertial(P_b_i_i, P_c_b_b, R_b_c, R_b_i, epsilon))
