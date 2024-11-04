import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

#optimisation
from skopt import gp_minimize
from scipy.linalg import solve_discrete_are

#EKF
from robot_localization_system import FilterConfiguration, Map, RobotEstimator

map = Map()
landmarks = map.landmarks  

# gloabl variables
W_range = 0.5 ** 2  
W_bearing = (np.pi * 0.5 / 180.0) ** 2 


coefficients_array = []
simulation_run = 0
speed = 0


def landmark_range_observations(base_position, W_range):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(W_range))
       
        y.append(range_meas)

    y = np.array(y)
    return y

def landmark_range_bearing(base_position, base_orientation):
    observations = []
    for landmark in landmarks:
        dx = landmark[0] - base_position[0]
        dy = landmark[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
        bearing_meas = wrap_angle(np.arctan2(dy, dx) - base_orientation) + np.random.normal(0, np.sqrt(W_bearing))
        observations.append((range_meas, bearing_meas))
    return observations


def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_


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
    global simulation_run
    # Initialize simulation
    conf_file_name = "robotnik.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    
    # Set floor friction, initial time, and other simulation parameters
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    time_step = sim.GetTimeStep()
    current_time = 0
    total_time_steps = 0

    # Initialize MPC
    num_states = 3
    num_controls = 2
    C = np.eye(num_states)
    
    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    
    # Update system matrices initially
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)
    
    # Set cost matrices
    regulator.setCostMatrices(Qcoeff, Rcoeff)
    
    u_mpc = np.zeros(num_controls)
    wheel_radius = 0.11
    wheel_base_width = 0.46
    cmd = MotorCommands()
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    
    pos_cutoff = 0.5
    bearing_cutoff = 0.3
    angular_cutoff = 3.0
    
    # Main loop
    while True:
        sim.Step(cmd, "torque")
        
        # Obtain measurements
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        y = landmark_range_observations(base_pos)
        
        # Update linearization points and regulator matrices
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        
        # Solve for optimal control sequence
        x0_mpc = np.hstack((base_pos[:2], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[0:num_controls]
        
        # Prepare and apply control command
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)
        
        # Check if the goal is reached
        position_good = np.abs(base_pos[0]) < pos_cutoff and np.abs(base_pos[1]) < pos_cutoff
        bearing_good = np.abs(base_bearing_) < bearing_cutoff or np.abs((3.14 - np.abs(base_bearing_)) < bearing_cutoff)
        angular_velocity_good = np.average(np.abs(angular_wheels_velocity_cmd)) < angular_cutoff

        # if position_good and bearing_good and angular_velocity_good:
        #     print("Goal reached")
        #     return total_time_steps

        if total_time_steps > 5000:
            simulation_run += 1
            print("\n\nSimulation run:", simulation_run, "\nbase_pos:", base_pos)
            return base_pos  # Penalize long simulations

        # Update time step
        current_time += time_step
        total_time_steps += 1

def objective_function(params):
    global coefficients_array
    # Extract parameters from the input array
    Q2, Q3, R1, R2, N_mpc = params
    
    # Set the cost matrices
    Qcoeff = np.array([Q2, Q2, Q3])
    Rcoeff = np.array([R1, R2])

    print("Qcoeff: ", Qcoeff)
    print("Rcoeff: ", Rcoeff)
    print("N_mpc: ", N_mpc)
    # Run the simulation with the provided parameters
    # step_count = run_simulation(Qcoeff, Rcoeff, N_mpc)
    base_pos = run_simulation(Qcoeff, Rcoeff, N_mpc)
    # coefficients_array.append([Q2, Q2, Q3, R1, R2, N_mpc, step_count])
    coefficients_array.append([Q2, Q2, Q3, R1, R2, N_mpc, base_pos[0], base_pos[1]])
    
    return base_pos[0] ** 2 + base_pos[1] ** 2 + base_pos[2] ** 2 # Return the square of the distance from the origin and the bearing


class EKF:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.x = initial_state  # State estimate
        self.P = initial_covariance  # State covariance
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

    def predict(self, control_input, A, B):
        # Prediction Step
        self.x = A @ self.x + B @ control_input  # Predicted state
        self.P = A @ self.P @ A.T + self.Q  # Predicted covariance

    def update(self, measurement, H, landmark_position):
        # Calculate expected range and bearing to the landmark
        dx = landmark_position[0] - self.x[0]
        dy = landmark_position[1] - self.x[1]
        expected_range = np.sqrt(dx**2 + dy**2)
        expected_bearing = wrap_angle(np.arctan2(dy, dx) - self.x[2])

        # Measurement residual (innovation)
        y = measurement - np.array([expected_range, expected_bearing])
        y[1] = wrap_angle(y[1])  # Ensure angle is within [-pi, pi]

        # Update Jacobian H with partial derivatives for range and bearing
        H[0, 0] = -dx / expected_range
        H[0, 1] = -dy / expected_range
        H[1, 0] = dy / (dx**2 + dy**2)
        H[1, 1] = -dx / (dx**2 + dy**2)

        # Calculate Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x += K @ y

        # More stable covariance update formula
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P @ (np.eye(len(self.x)) - K @ H).T + K @ self.R @ K.T

def P_(A, B, Q, R):
    P = np.eye(A.shape[0])
    for i in range(100):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return P

def main():
    global coefficients_array
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0
    total_time_steps = 0
   
    

    # res = gp_minimize(
    #     objective_function,
    #     dimensions=[(0, 400), (-1000,1000), (0.2, 1.0), (0.2, 1.0), (2, 12)],
    #     n_calls=100,  # Number of evaluations
    #     n_random_starts=10,  # Start with 10 random evaluations
    # )

    # optimal_parameters = res.x  # This gives [Q1, Q2, Q3, R, N]
    # all_results = res.func_vals

    # np.save("optimal_parameters_distance2.npy", optimal_parameters)
    # np.save("all_results_distance2.npy", all_results)


    optimal_parameters = np.load("optimal_parameters_distance.npy")

    print("Optimal parameters: ", optimal_parameters)

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = int(optimal_parameters[-1])
    # N_mpc = 10 # optimal with P included
    #[Q1, Q2, Q3, R, N]
    #[991, 36, 628, 0.7761416026955681, 12] for 500 iterations, position_cutoff = 0.5, bearing_cutoff = 0.3, angular_cutoff = 3.0
    #[784, 1000, 1.0, 12] for 300 iterations, position_cutoff = 0.5, bearing_cutoff = 0.3, angular_cutoff = 2.0
    #[378, 378, 472, 0.5, 9] for 300 iterations, position_cutoff = 0.5, bearing_cutoff = 0.3, angular_cutoff = 2.0 (state space change to 400, 400, 500)

    #[Q1, Q2, Q3, R1, R2, N]

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    # # update A,B,C matrices
    # # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # # you can linearize around the final state and control of the robot (everything zero)
    # # or you can linearize around the current state and control of the robot
    # # in the second case case you need to update the matrices A and B at each time step
    # # and recall everytime the method updateSystemMatrices
    init_pos  = np.array([2.0, 3.0, 0.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    
    # Initialize data storage
    # EKF Initialization
    # initial_covariance = np.eye(3)
    # process_noise = np.diag([0.02, 0.02, 0.02])
    # measurement_noise = np.diag([0.3, 0.3])

    # ekf = EKF(init_pos, initial_covariance, process_noise, measurement_noise)
    base_pos_all, base_bearing_all = [], []
    x_true_history, x_est_history = [], []
    
    # Define the cost matrices
    Qcoeff = np.array([optimal_parameters[0], optimal_parameters[0], optimal_parameters[1]])
    Rcoeff = np.array([optimal_parameters[2], optimal_parameters[2]])
    # Qcoeff = np.array([310, 310, 80.0])
    # Rcoeff = np.array([0.5, 0.5])
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   

    u_mpc = np.zeros(num_controls)

    true_positions = []
    estimated_positions = []
    position_errors = []
    bearing_errors = []

    # ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    #### conditions for robot reaching the goal ####
    pos_cutoff = 0.5
    bearing_cutoff = 0.3
    angular_cutoff = 2.0

    
    while True:
        # True state propagation (with process noise)
        # ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Kalman filter prediction
       
    
        # Get the measurements from the simulator ###########################################
        #  # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        # measurements = landmark_range_observations(base_pos_no_noise,W_range)

    
        # # EKF Prediction
        # ekf.predict(u_mpc, regulator.A, regulator.B)

        # # EKF Update using landmarks
        # # measurements = landmark_range_bearing(base_pos, base_bearing_)
        # for idx, landmark in enumerate(landmarks):
        #     H = np.zeros((2, 3))
        #     measurement = np.array(measurements[idx])
        #     ekf.update(measurement, H, landmark)

        # # Use EKF estimated state for control
        # x_estimated_for_mpc = ekf.x

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
       
   
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        # cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        # cur_u_for_linearization = u_mpc
        cur_state_x_for_linearization = [0,0,0]
        cur_u_for_linearization = [0,0]
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        # regulator.updateSystemMatrices(sim, x_estimated_for_mpc, u_mpc)

        #find P matrix
        # P = solve_discrete_are(regulator.A, regulator.B, regulator.Q, regulator.R)
        P = None
        print("P matrix:", P)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std(P)
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar, P)
        
        x0_mpc = np.hstack((base_pos[:2], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        print("u_mpc",np.abs(u_mpc))
        print("base_pos",base_pos)
        print("base_bearing_",base_bearing_)
        print("angular_wheels_velocity_cmd",angular_wheels_velocity_cmd)

        ## Check if the goal is reached ##

        # position_good = True if np.abs(base_pos[0]) < pos_cutoff and np.abs(base_pos[1]) < pos_cutoff else False
        # bearing_good = True if np.abs(base_bearing_) < bearing_cutoff or np.abs((3.14 - np.abs(base_bearing_)) < bearing_cutoff) else False
        # angular_velocity_good = True if np.average(np.abs(angular_wheels_velocity_cmd)) < angular_cutoff else False

        #condition for robot reaching the goal
        # if position_good and bearing_good and angular_velocity_good:
        #     print("Goal reached")
        #     break

        if total_time_steps > 10000:
            print("Time limit reached")
            break


        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # # Store data for plotting if necessary
        base_pos_all.append(base_pos_no_noise)
        base_bearing_all.append(base_bearing_no_noise_ - 3.14)
        # x_true_history.append([base_pos[0], base_pos[1], base_bearing_])
        # x_est_history.append(ekf.x)
        position_errors.append(np.sqrt((base_pos[0]) ** 2 + (base_pos[1]) ** 2))
        bearing_errors.append(np.abs(base_bearing_))

        # # Update current time
        current_time += time_step
        total_time_steps += 1

        print(total_time_steps)

    # print coefficients_array
    print("Coefficients array: ", coefficients_array)
    print("Optimal parameters: ", optimal_parameters)
    # print("All results: ", all_results)

    #make array of x true history and x est history
    # x_true_history = np.array(x_true_history)
    # x_est_history = np.array(x_est_history)

    # # Save the coefficients array and optimal parameters
    # coefficients_array = np.array(coefficients_array)
    # np.save("coefficients_array.npy", coefficients_array)
    # np.save("optimal_parameters.npy", optimal_parameters)
    # np.save("all_results.npy", all_results)
    

    #plotting
    # coefficients_array = np.array(coefficients_array)
    # plt.figure()
    # plt.plot(coefficients_array[:, -1], 'r-', label='Time Steps')
    # plt.xlabel('Iteration')
    # plt.ylabel('Time Steps')
    # plt.title('Time Steps')
    # plt.legend()
    # plt.grid()
    # plt.show()

    position_errors = np.array(position_errors)

    #plot the distance from the origin the robot is at each time step
    # plt.figure()
    # plt.plot(position_errors, 'r-', label='Position Error')
    # # plot the final poisition error
    # plt.plot(position_errors[-1], label='Final Distance from Goal = {:.2f}cm'.format(position_errors[-1]))
    # plt.plot(bearing_errors[-1], label='Final Bearing Error from Goal = {:.2f}rad'.format(bearing_errors[-1]))
    # plt.xlabel('Time Step')
    # plt.ylabel('Position Error')
    # plt.title('Position Error')
    # plt.legend()
    # plt.grid()
    # plt.show()




    # Plotting 
    #add visualization of final x, y, trajectory and theta
    base_pos_all = np.array(base_pos_all)
    base_bearing_all = np.array(base_bearing_all)
    # plt.figure()
    # plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], 'b-', label='Robot Trajectory')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.title('Robot Trajectory')
    # plt.legend()
    # plt.grid()
    # plt.show()


    # plt.figure()
    # plt.plot(np.arctan2(np.sin(base_bearing_all), np.cos(base_bearing_all)), 'r-', label='Robot Bearing')
    # plt.xlabel('Time Step')
    # plt.ylabel('Bearing [rad]')
    # plt.title('Robot Bearing')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Plot Trajectories
    # plt.figure()
    # plt.plot(x_true_history[:, 0], x_true_history[:, 1], label="True Path")
    # plt.plot(x_est_history[:, 0], x_est_history[:, 1], linestyle='--', label="EKF Estimated Path")
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], color='red', marker='x', label="Landmarks")
    # plt.xlabel("X Position [m]")
    # plt.ylabel("Y Position [m]")
    # plt.legend()
    # plt.grid()

    # for i, label in enumerate(['X', 'Y', 'Theta']):
    #     plt.figure()
    #     plt.plot(x_true_history[:, i], label=f"True {label}")
    #     plt.plot(x_est_history[:, i], linestyle='--', label=f"Estimated {label}")
    #     plt.xlabel("Time step")
    #     plt.ylabel(label)
    #     plt.legend()
    #     plt.grid()

    # plt.show()



if __name__ == '__main__':
    main()