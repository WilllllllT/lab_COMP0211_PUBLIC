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

def landmark_bearing_observations(base_position, base_bearing):
    y = []
    for lm in landmarks:
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        bearing_true = np.arctan2(dy, dx) - base_bearing
        bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))
        # 将角度限制在 [-π, π]
        bearing_meas = np.arctan2(np.sin(bearing_meas), np.cos(bearing_meas))
        y.append(bearing_meas)
    return np.array(y)


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
   
    ## run the bayes optimisation ##

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


    optimal_parameters = np.load('Saved_np\optimal_parameters_distance.npy')

    print("Optimal parameters: ", optimal_parameters)

    # initializing MPC
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    # N_mpc = int(optimal_parameters[-1])
    N_mpc = 4 # optimal with P included


    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    init_pos  = np.array([2.0, 3.0, 0.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    
    # Set the cost matrices
    base_pos_all, base_bearing_all = [], []
    true_positions, estimated_positions = [], []
    
    # Define the cost matrices
    Qcoeff = np.array([optimal_parameters[0], optimal_parameters[0], optimal_parameters[1]])
    Rcoeff = np.array([optimal_parameters[2], optimal_parameters[2]])
    # Qcoeff = np.array([310, 310, 80.0])
    # Rcoeff = np.array([0.5, 0.5])
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   

    u_mpc = np.zeros(num_controls)

    # Initialize the EKF
    filter_config = FilterConfiguration()
    map = Map()
    estimator = RobotEstimator(filter_config, map)
    estimator.start()


    position_errors, bearing_errors, true_positions, estimated_positions = [], [], [] ,[]
    

    # ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
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

        # Range and bearing observations
        y_range = landmark_range_observations(base_pos_no_noise, W_range)
        y_bearing = landmark_bearing_observations(base_pos, base_bearing_)

        # EKF 预测和更新
        estimator.set_control_input(u_mpc)
        estimator.predict_to(current_time)
        estimator.update_from_landmark_observations(y_range, y_bearing)

        # 获取当前状态估计
        x_est, Sigma_est = estimator.estimate()

        # Store true and estimated positions
        true_positions.append(base_pos_no_noise[:2]) 
        estimated_positions.append(x_est[:2]) 
    

        # Compute the matrices needed for MPC optimization
        cur_state_x_for_linearization = [x_est[0], x_est[1], x_est[2]]
        regulator.updateSystemMatrices(sim, x_est, u_mpc)
        # cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        # cur_u_for_linearization = u_mpc
        # regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        # regulator.updateSystemMatrices(sim, x_estimated_for_mpc, u_mpc)

        #find P matrix

        P = solve_discrete_are(regulator.A - 0.05*np.eye(num_states), regulator.B, regulator.Q, regulator.R)
        # P = None
        print("P matrix:", P)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std(P)
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        
        x0_mpc = np.hstack((base_pos[:2], base_bearing_))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        # u_mpc = -H_inv @ F @ x0_mpc # without EKF

        u_mpc = -H_inv @ F @ x_est # with EKF
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)


        sim.Step(cmd, "torque")
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
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_ - 3.14)
        # x_true_history.append([base_pos[0], base_pos[1], base_bearing_])
        # x_est_history.append(ekf.x)
        position_errors.append(np.sqrt((base_pos[0]) ** 2 + (base_pos[1]) ** 2))
        bearing_errors.append(np.abs(base_bearing_))

        # # Update current time
        current_time += time_step
        total_time_steps += 1

        print(total_time_steps)

    ############ EKF PLOTTING ################
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)

    # 绘制真实轨迹和估计轨迹的对比图
    plt.figure()
    plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Trajectory', color='blue')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Trajectory', color='orange', linestyle='--')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Robot Trajectory Comparison (Total Time: {current_time})')
    plt.legend()
    plt.show()

    # 打印初始位置和状态信息
    print("Initial Base Position:", sim.GetBasePosition())
    print("Initial Base Orientation:", sim.GetBaseOrientation())
    print(f"Final Base Position: {base_pos}, Bearing: {base_bearing_}, u_mpc: {u_mpc}")
    ############################################

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

    # plot base_pos_all
    

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