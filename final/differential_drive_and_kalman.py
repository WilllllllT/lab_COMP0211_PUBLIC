import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller, regulation_polar_coordinates, regulation_polar_coordinate_quat, wrap_angle, velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model_copy import RegulatorModel
import robot_localization_system as RE
from scipy.linalg import solve_discrete_are

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
    [5, 10],
    [15, 5],
    [10, 15]
])


def landmark_range_observations(base_position):
    y = []
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        y.append(range_meas)
    return np.array(y)


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized
    rot_quat = quat.toRotationMatrix()
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Extract roll, pitch, yaw
    return base_euler[2]  # Yaw as bearing


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def run_simulation(Qcoeff, Rcoeff, N_mpc):
    # Initialize simulation
    conf_file_name = "robotnik.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # Initialize the Kalman Filter parameters
    initial_state_estimate = np.array([2.0, 3.0])  # Initial position and bearing estimate
    initial_covariance_estimate = np.diag([1.0, 1.0, 0.5]) ** 2  # Initial covariance estimate
    # Kalman Filter process and measurement noise
    process_noise_covariance = np.eye(3) * 0.01  # Process noise covariance
    measurement_noise_covariance = np.eye(len(landmarks)) * W_range  # Measurement noise covariance



    # Set initial state and covariance for Kalman Filter
    x_est, P_est = initial_state_estimate, initial_covariance_estimate


    # Initialize MPC
    num_states = 3
    num_controls = 2
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    regulator.setCostMatrices(Qcoeff, Rcoeff)
    wheel_radius = 0.11
    wheel_base_width = 0.46
    cmd = MotorCommands()
    u_mpc = np.zeros(num_controls)


    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    A, B = regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    

    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    

    pos_cutoff = 0.5
    bearing_cutoff = 0.3
    angular_cutoff = 3.0
    total_time_steps = 0

    while True:
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        
        # #shapes of x_est, u_mpc, A, B
        # print(x_est.shape)
        # print(u_mpc.shape)
        # print(A.shape)
        # print(B.shape)

        # Kalman filter prediction
        x_pred, P_pred = regulator.kalman_filter_predict(x_est, P_est, np.zeros(num_controls), process_noise_covariance)
        

        # Get noisy measurements and update Kalman filter
        y = landmark_range_observations(base_pos)
        x_est, P_est = regulator.kalman_filter_update(x_pred, P_pred, y, landmarks, measurement_noise_covariance, base_pos)

        # MPC based on estimated state (Kalman Filter output)
        x0_mpc = x_est.flatten()
        regulator.updateSystemMatrices(sim, x0_mpc, np.zeros(num_controls))
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc[:3]
        u_mpc = u_mpc[0:num_controls]
        
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["velocity"] * 4)

        # Check if goal is reached
        position_good = np.linalg.norm(base_pos[:2]) < pos_cutoff
        bearing_good = abs(base_bearing_) < bearing_cutoff or abs(3.14 - abs(base_bearing_)) < bearing_cutoff
        angular_velocity_good = np.mean(np.abs(angular_wheels_velocity_cmd)) < angular_cutoff

        if position_good and bearing_good and angular_velocity_good:
            print("Goal reached")
            return total_time_steps

        if total_time_steps > 10000:
            print("Time limit reached")
            return 10000  # Penalize long simulations

        total_time_steps += 1


def main():
    # Define MPC cost coefficients
    Qcoeff = np.array([378, 378, 472])
    Rcoeff = np.array([0.5, 0.5])
    N_mpc = 9  # Horizon length
    
    # Run the simulation with Kalman Filter and MPC
    run_simulation(Qcoeff, Rcoeff, N_mpc)
    

if __name__ == '__main__':
    main()
