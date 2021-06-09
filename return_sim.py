import numpy as np
#from scipy import linalg as la
import time
import matplotlib.pyplot as plt
import math
import random

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Linear Quadratic Regulator example
#   (two-wheeled differential drive robot car)

######################## DEFINE CONSTANTS #####################################
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)

# Optional Variables
#max_linear_velocity = 3.0 # meters per second
#max_angular_velocity = 1.5708 # radians per second
max_angle = 0.610 # radians per second
max_angle_change = 1 #dummy for now
mass = 682.04 #kg
max_thrust = 587.16 #N
avg_drag_force = 59.1613 #N
drag_acceleration = avg_drag_force / mass
max_acceleration = max_thrust / mass
dt = 1.0
o_noises = []
p_noises = []
fin_noises = []
thrust_noises = []
controls = []



def getB(alpha, beta, gamma):
        """
        Expresses how the state of the system changes
        from t-1 to t due to the control commands (i.e. control inputs).
        """
        B = np.zeros((9,4))
        yaw_rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
                            [np.sin(alpha), np.cos(alpha), 0.0],
                            [0.0, 0.0, 1.0]])
        pitch_rotation = np.array([[np.cos(beta), 0.0, -np.sin(beta)],
                            [0.0, 1.0, 0.0],
                            [np.sin(beta), 0.0, np.cos(beta)]])
        roll_rotation = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(gamma), -np.sin(gamma)],
                            [0.0, np.sin(gamma), np.cos(gamma)]])
        rotation_matrix = yaw_rotation @ pitch_rotation @ roll_rotation
        #print(rotation_matrix[0][0])
        #print(np.cos(beta)*np.cos(alpha))
        #print(rotation_matrix[1][0])
        #print(np.cos(beta)*np.sin(alpha))
        #print(rotation_matrix[2][0])
        #print(np.sin(beta))
        #rotation_matrix[0][0] = np.cos(beta)*np.cos(alpha)
        #rotation_matrix[1][0] = np.cos(beta)*np.sin(alpha)
        #rotation_matrix[2][0] = np.sin(beta)
        B[0][0] = 0.5*rotation_matrix[0][0]*dt*dt
        B[1][0] = rotation_matrix[0][0]*dt
        B[2][0] = 0.5*rotation_matrix[1][0]*dt*dt
        B[3][0] = rotation_matrix[1][0]*dt
        B[4][0] = 0.5*rotation_matrix[2][0]*dt*dt
        B[5][0] = rotation_matrix[2][0]*dt
        B[6][1] = 0.2*dt #0.07
        B[7][2] = 0.1*dt #0.035
        B[7][3] = 0.1*dt #0.035
        B[8][2] = 0.2*dt
        B[8][3] = -0.2*dt
        return B

def clip_and_noise(control_input_t_minus_1, last_rud, last_lf, last_rf):
    #clipped acceleration is 1.09 m/s^2 based on maximum available thrust at average speed of 2 knots (1 m/s)


    fin_noise = np.random.normal(0, 0.02*max_angle, 1)
    thrust_noise = np.random.normal(0, 0.02*max_acceleration, 1)
    fin_noises.append(fin_noise[0])
    thrust_noises.append(thrust_noise[0])

    control_input_t_minus_1[0] += thrust_noise
    control_input_t_minus_1[1] += fin_noise
    control_input_t_minus_1[2] += fin_noise
    control_input_t_minus_1[3] += fin_noise
    clipped = [0, 0, 0, 0]
    clipped[0] = np.clip(control_input_t_minus_1[0], -max_acceleration, max_acceleration)
    clipped[1] = np.clip(control_input_t_minus_1[1], -max_angle_change + last_rud, max_angle_change + last_rud)
    clipped[2] = np.clip(control_input_t_minus_1[2], -max_angle_change + last_lf, max_angle_change + last_lf)
    clipped[3] = np.clip(control_input_t_minus_1[3], -max_angle_change + last_rf, max_angle_change + last_rf)

    clipped[1] = np.clip(control_input_t_minus_1[1], -max_angle, max_angle)
    clipped[2] = np.clip(control_input_t_minus_1[2], -max_angle, max_angle)
    clipped[3] = np.clip(control_input_t_minus_1[3], -max_angle, max_angle)

    clipped[0] -= drag_acceleration
    return np.asarray(clipped)

def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    # These next 6 lines of code which place limits on the angular and linear
    # velocities of the robot car can be removed if you desire.
    #control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
    #                                                                        -max_linear_velocity,
    #                                                                        max_linear_velocity)
    #control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
    #                                                                        -max_angular_velocity,
    #                                                                        max_angular_velocity)
    position_noise = np.random.normal(0, 0.08, 1) #meters
    orientation_noise = np.random.normal(0, 0.06, 1) #radians
    o_noises.append(orientation_noise[0])
    p_noises.append(position_noise[0])

    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)
    state_estimate_t[0] += position_noise
    state_estimate_t[2] += position_noise
    state_estimate_t[4] += position_noise
    state_estimate_t[6] += orientation_noise
    state_estimate_t[7] += orientation_noise
    state_estimate_t[8] += orientation_noise
    return state_estimate_t

def lqr(actual_state_x, desired_state_xf, Q, R, A, B):
    """
    Discrete-time linear quadratic regulator for a nonlinear system.

    Compute the optimal control inputs given a nonlinear system, cost matrices,
    current state, and a final state.

    Compute the control variables that minimize the cumulative cost.

    Solve for P using the dynamic programming method.

    :param actual_state_x: The current state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param desired_state_xf: The desired state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param dt: The size of the timestep in seconds -> float

    :return: u_star: Optimal action u for the current state
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
    """
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf

    # Solutions to discrete LQR problems are obtained using the dynamic
    # programming method.
    # The optimal solution is obtained recursively, starting at the last
    # timestep and working backwards.
    # You can play with this number
    N = 50

    # Create a list of N + 1 elements
    P = [None] * (N + 1)

    Qf = Q

    # LQR via Dynamic Programming
    P[N] = Qf

    # For i = N, ..., 1
    for i in range(N, 0, -1):

        # Discrete-time Algebraic Riccati equation to calculate the optimal
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)
        #print("P[i]: ")
        #print(P[i-1])

    # Create a list of N elements
    K = [None] * N
    u = [None] * N

    # For i = 0, ..., N - 1
    for i in range(N):

        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
        #print("K[i]: ")
        #print(K[i])
        u[i] = K[i] @ x_error
        #print("u[i]: ")
        #print(u[i])
    #print("x_error: ")
    #print(x_error)

    # Optimal control input is u_star
    u_star = u[N-1]

    #P = la.solve_discrete_are(A, B, Q, R)
    #K = la.solve(R + B.T.dot(P).dot(B), B.T.dot(P).dot(A))
    #u_star = K.dot(x_error)
    #K, S, E = control.lqr(A, B, Q, R)
    #u_star = K.dot(x_error)
    return u_star

def main():

    # Actual state
    actual_state_x = np.array([0,0,0,0,0,0,0,0,0])

    xpos = []
    ypos = []
    zpos = []

    rudderpos = [0.0]
    lfpos = [0.0]
    rfpos = [0.0]



    # Desired state [x, y, z, azimuth angle, elevation angle, tilt angle]
    # [meters, meters, meters, radians, radians, radians]
    desx = 100.0
    desdx = 0.0
    desy = 100.0
    desdy = 0.0
    desz = -100.0
    desdz = 0.0
    desaz = np.arctan(desy/desx)
    desel = np.arctan(desz/(np.sqrt(desx**2+desy**2)))
    destl = 0.0
    desired_state_xf = np.array([desx,desdx,desy,desdy,desz,desdz,desaz,desel,destl])
    # A matrix
    # 3x3 matrix -> number of states x number of states matrix
    # Expresses how the state of the system [x,y,yaw] changes
    # from t-1 to t when no control command is executed.
    # Typically a robot on wheels only drives when the wheels are told to turn.
    # For this case, A is the identity matrix.
    # Note: A is sometimes F in the literature.
    A = np.array([[1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    # R matrix
    # The control input cost matrix
    # Experiment with different R matrices
    # This matrix penalizes actuator effort (i.e. rotation of the
    # motors on the wheels that drive the linear velocity and angular velocity).
    # The R matrix has the same number of rows as the number of control
    # inputs and same number of columns as the number of
    # control inputs.
    # This matrix has positive values along the diagonal and 0s elsewhere.
    # We can target control inputs where we want low actuator effort
    # by making the corresponding value of R large.
    R = np.array([[750.00, 0.00, 0.00, 0.00],  # Penalty for thrust effort
                  [0.00, 0.01, 0.00, 0.00],  # Penalty for rudder effort
                  [0.00, 0.00, 0.01, 0.00],  # Penalty for left fin effort
                  [0.00, 0.00, 0.00, 0.01]]) # Penalty for right fin effort

    # Q matrix
    # The state cost matrix.
    # Experiment with different Q matrices.
    # Q helps us weigh the relative importance of each state in the
    # state vector (X, Y, YAW ANGLE).
    # Q is a square matrix that has the same number of rows as
    # there are states.
    # Q penalizes bad performance.
    # Q has positive values along the diagonal and zeros elsewhere.
    # Q enables us to target states where we want low error by making the
    # corresponding value of Q large.
    Q = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize X position error
                  [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Penalize dX error
                  [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize Y position error
                  [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # Penalize dY error
                  [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0], # Penalize Z position error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], # Penalize dZ error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],  # Penalize AZIMUTH heading error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], # Penalize ELEVATION heading error
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]) # Penalize TILT heading error

    # Launch the robot, and have it move to the desired goal destination
    old_mag = 300.0
    for i in range(360):
        print(f'iteration = {i} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')

        state_error = actual_state_x - desired_state_xf
        concerns = np.array([state_error[0], state_error[2], state_error[4]])
        state_error_magnitude = np.linalg.norm(state_error)
        print(f'State Error Magnitude = {state_error_magnitude}')
        concerns_magnitude = np.linalg.norm(concerns)
        print(f'Position Error Magnitude = {concerns_magnitude}')
        B = getB(actual_state_x[6], actual_state_x[7], actual_state_x[8])

        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x, desired_state_xf, Q, R, A, B)
        clipped_control_input = clip_and_noise(optimal_control_input, rudderpos[-1], lfpos[-1], rfpos[-1])
        print(f'Control Input = {optimal_control_input}')
        print(f'Clipped Control = {clipped_control_input}')
        controls.append(clipped_control_input.tolist())

        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, optimal_control_input)
        xpos.append(actual_state_x[0])
        ypos.append(actual_state_x[2])
        zpos.append(actual_state_x[4])

        rudderpos.append(clipped_control_input[1])
        lfpos.append(clipped_control_input[2])
        rfpos.append(clipped_control_input[3])

        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 1.0 or concerns_magnitude < 3.8:
            print("\nGoal Has Been Reached Successfully!")
            break
        if state_error_magnitude > old_mag+0.2:
            break
        old_mag = state_error_magnitude
        print()


    ax = plt.axes(projection='3d')
    ax.view_init(azim=45, elev=30)
    # ax.plot3D(obstacle_xs, obstacle_ys, obstacle_zs, 'ro', alpha=0.3, label='obstacles')
    # ax.plot3D(xs,ys,zs, color='black', label='planned path (waypoints)')
    ax.plot3D(abs(xpos[-1]-desx), abs(ypos[-1]-desy), abs(zpos[-1]-desz), 'b*', label='plan end')
    #ax.plot3D([desx], [desy], [desz], 'g*', label='plan end')
    ax.plot3D(xpos,ypos,zpos,'g',label='2nd order path')
    ax.plot3D([xpos[0]], [ypos[0]], [zpos[0]], 'r*', label = '2nd order end')
    ax.plot3D([xpos[-1]], [ypos[-1]], [zpos[-1]], 'm*', label = '2nd order start')
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('y (meters)')
    ax.set_zlabel('z (meters)')
    plt.title(f'2nd order Control')
    plt.legend()
    plt.show()

    tim = [i for i in range(len(controls))]
    accs = [controls[i][0] for i in range(len(controls))]
    #st2 = f'trial{n+1}_acc_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration control (m/s^2)')
    plt.title('LQR Acceleration Control Input')
    plt.plot(tim, accs)
    plt.show()
    plt.clf()

    #tim = [i for i in range(len(realcontrol))]
    ruds = [controls[i][1] for i in range(len(controls))]
    #st3 = f'trial{n+1}_rud_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rudder control (radians)')
    plt.title('LQR Rudder Control Input')
    plt.plot(tim, ruds)
    plt.show()
    plt.clf()

    #tim = [i for i in range(len(realcontrol))]
    lfs = [controls[i][2] for i in range(len(controls))]
    #st4 = f'trial{n+1}_lf_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Left Fin control (radians)')
    plt.title('LQR Left Fin Control Input')
    plt.plot(tim, lfs)
    plt.show()
    plt.clf()

    tim = [i for i in range(len(controls))]
    rfs = [controls[i][3] for i in range(len(controls))]
    #st5 = f'trial{n+1}_rf_control_updated'
    plt.xlabel('Time (seconds)')
    plt.ylabel('Right Fin control (radians)')
    plt.title('LQR Right Fin Control Input')
    plt.plot(tim, rfs)
    plt.show()
    plt.clf()

    plt.xlabel('Orientation Noises (radians)')
    plt.ylabel('Occurences')
    plt.title('Orientation Noises')
    #plt.plot(tim, o_noises)
    plt.hist(o_noises, bins=20)
    plt.show()
    plt.clf()

    plt.xlabel('Position Noises (meters)')
    plt.ylabel('Occurences')
    plt.title('Position Noises')
    #plt.plot(tim, p_noises)
    plt.hist(p_noises, bins=20)
    plt.show()
    plt.clf()

    plt.xlabel('Thrust Noises (radians)')
    plt.ylabel('Occurences')
    plt.title('Thrust Noises')
    #plt.plot(tim, thrust_noises)
    plt.hist(thrust_noises, bins=20)
    plt.show()
    plt.clf()

    plt.xlabel('Fin/Rudder Noises (radians)')
    plt.ylabel('Occurences')
    plt.title('Fin and Rudder Noises')
    #plt.plot(tim, fin_noises)
    plt.hist(fin_noises, bins=20)
    plt.show()
    plt.clf()



# Entry point for the program
main()
