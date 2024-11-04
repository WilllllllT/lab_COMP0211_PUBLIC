from pprint import pp
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from regulator_model import RegulatorModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    

def getSystemMatrices(sim, num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    time_step = sim.GetTimeStep()

    # Define the system matrices

    # Initialize A and B matrices
    A = np.eye(num_states) # State transition matrix, remain the same if no control input
    B = np.zeros((num_states, num_controls)) # Control input matrix, should refresh every time step

    # Fill in the A matrix based on the state evolution equations
    for i in range(num_joints):
        A[i, num_joints + i] = time_step  # q_k+1 = q_k + dot_q_k * delta_t
        A[num_joints + i, num_joints + i] = 1  # dot_q_k+1 = dot_q_k + dot_q_k * delta_t

        if damping_coefficients is not None:
            A[num_joints + i, num_joints + i] -= time_step * damping_coefficients[i]
    
    # Fill in the B matrix
    for i in range(num_joints):
        B[num_joints + i, i] = time_step  # Control input affects velocity state

    
    return A, B
    
    # TODO: Finish the system matrices 


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # # Q = 1 * np.eye(num_states)  # State cost matrix
    # Q = 1000 * np.eye(num_states) # Test with parameters with 1000, 10000, 100000
    # Q[num_joints:, num_joints:] = 0.0 # Penalize only the position states, not on velocity states
    
    # # Define R matrix for penalizing control inputs (accerlations)
    # R = 0.1 * np.eye(num_controls)  # Control input cost matrix

    pp = 1000  # Position penalty weight
    vp = 0.0  # Velocity penalty weight
    ip = 0.1  # Control input penalty weight

    # Define Q matrix to penalize position and velocity components
    Q = np.zeros((num_states, num_states))
    Q[:num_joints, :num_joints] = pp * np.eye(num_joints)  # Adjust the penalty for position component with pp
    Q[num_joints:, num_joints:] = vp * np.eye(num_joints)  # Adjust the penalty for velocity component with vp
    
    # Define R matrix to penalize control inputs
    R = ip * np.eye(num_controls) # Adjust the penalty for control input with ap
    
    return Q, R


def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    torque_all, acc_all = [], [] # Store torque and acceleration

    # Define the matrices
    A, B = getSystemMatrices(sim, num_joints)
    Q, R = getCostMatrices(num_joints)

    # # Display the system matrices and cost matrices for verification
    # (A, B, Q, R)
    # print(f"A matrix: {A}")
    # print(f"B matrix: {B}")
    # print(f"Q matrix: {Q}")
    # print(f"R matrix: {R}")
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states)
    # Compute the matrices needed for MPC optimization
    S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
    H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
    
    # Main control loop
    episode_duration = 5
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    sim.ResetPose()
    # sim.SetSpecificPose([1, 1, 1, 0.4, 0.5, 0.6, 0.7])
    
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_joints]
       
        # Control command
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # print(cmd.tau_cmd)
        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes) # Store measured position
        qd_mes_all.append(qd_mes) # Store measured velocity
        torque_all.append(cmd.tau_cmd) # Store torque
        acc_all.append(u_mpc) #Store input acceleration


        # time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        # print(f"Current time: {current_time}")
    
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(12, 5))
        
        # Position plot for joint i
        plt.subplot(2, 2, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()
        
        # Velocity plot for joint i
        plt.subplot(2, 2, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()
        
        # Torque plot for joint i
        plt.subplot(2, 2, 3)
        plt.plot([tau[i] for tau in torque_all], label=f'Torque - Joint {i+1}')
        plt.title(f'Torque for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Torque')
        plt.legend()
        
        # Acceleration plot for joint i
        plt.subplot(2, 2, 4)
        plt.plot([a[i] for a in acc_all], label=f'Acceleration - Joint {i+1}')
        plt.title(f'Acceleration for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Acceleration')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()