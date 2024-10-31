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




# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])

coefficients_array = []


def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y


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

        if position_good and bearing_good and angular_velocity_good:
            print("Goal reached")
            return total_time_steps

        if total_time_steps > 10000:
            print("Time limit reached")
            return 10000  # Penalize long simulations

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
    step_count = run_simulation(Qcoeff, Rcoeff, N_mpc)
    coefficients_array.append([Q2, Q2, Q3, R1, R2, N_mpc, step_count])
    
    return step_count


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
   
    # Initialize data storage
    base_pos_all, base_bearing_all = [], []

    # res = gp_minimize(
    #     objective_function,
    #     dimensions=[(0, 400), (-500, 500), (0.5, 1.0), (0.5, 1.0), (2, 12)],
    #     n_calls=500,  # Number of evaluations
    #     n_random_starts=30,  # Start with 10 random evaluations
    # )

    # optimal_parameters = res.x  # This gives [Q1, Q2, Q3, R, N]
    # all_results = res.func_vals

    optimal_parameters = np.load("optimal_parameters_2.npy")

    print("Optimal parameters: ", optimal_parameters[0])

    # initializing MPC
     # Define the matrices
    num_states = 3
    num_controls = 2
   
    
    # Measuring all the state
    
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = int(optimal_parameters[-1])
    # N_mpc = 9
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
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    
    
    # Define the cost matrices
    Qcoeff = np.array([optimal_parameters[0], optimal_parameters[0], optimal_parameters[1]])
    Rcoeff = np.array([optimal_parameters[2], optimal_parameters[2]])
    # Qcoeff = np.array([378, 378, 472])
    # Rcoeff = np.array([0.5, 0.5])
    regulator.setCostMatrices(Qcoeff,Rcoeff)
   

    u_mpc = np.zeros(num_controls)

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

    # Solve for the terminal cost matrix P TASK 2
    P = solve_discrete_are(regulator.A, regulator.B, np.diag(Qcoeff), np.diag(Rcoeff))
    print(P)

    
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
        y = landmark_range_observations(base_pos)

    
        # Update the filter with the latest observations
        
    
        # Get the current state estimate
        

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
       
   
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)


        H[-P.shape[0]:, -P.shape[1]:] += P


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

        position_good = True if np.abs(base_pos[0]) < pos_cutoff and np.abs(base_pos[1]) < pos_cutoff else False
        bearing_good = True if np.abs(base_bearing_) < bearing_cutoff or np.abs((3.14 - np.abs(base_bearing_)) < bearing_cutoff) else False
        # velocity_good = True if (np.abs(16 - u_mpc[0]) < velocity_cutoff and np.abs(u_mpc[1]) < velocity_cutoff) or (np.abs(u_mpc[0]) < velocity_cutoff and np.abs(16 - u_mpc[1]) < velocity_cutoff) else False
        angular_velocity_good = True if np.average(np.abs(angular_wheels_velocity_cmd)) < angular_cutoff else False

        #condition for robot reaching the goal
        if position_good and bearing_good and angular_velocity_good:
            print("Goal reached")
            break

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
        base_bearing_all.append(np.arctan2(np.sin(base_bearing_), np.cos(base_bearing_)))

        # # Update current time
        current_time += time_step
        total_time_steps += 1

        print(total_time_steps)

    # print coefficients_array
    # print("Coefficients array: ", coefficients_array)
    # print("Optimal parameters: ", optimal_parameters)
    # print("All results: ", all_results)

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


    # Plotting 
    #add visualization of final x, y, trajectory and theta
    base_pos_all = np.array(base_pos_all)
    base_bearing_all = np.array(base_bearing_all)
    plt.figure()
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], 'b-', label='Robot Trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(base_bearing_all, 'r-', label='Robot Bearing')
    plt.xlabel('Time Step')
    plt.ylabel('Bearing [rad]')
    plt.title('Robot Bearing')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()