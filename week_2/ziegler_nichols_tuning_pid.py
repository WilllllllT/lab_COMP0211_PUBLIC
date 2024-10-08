import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir, use_gui=False)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")



# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd = np.array([0]*dyn_model.getNumberofActuatedJoints())
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

   
    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors


    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    

    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    if plot:
        plt.figure()
        plt.plot(np.array(q_mes_all)[:, joints_id], label="Measured Position")
        plt.plot(np.array(q_d_all)[:, joints_id], label="Desired Position")
        plt.xlabel('Time step')
        plt.ylabel('Joint Position')
        plt.title(f"Joint {joints_id} Position vs Desired Position (Kp={kp})")
        plt.legend()
        plt.show()
    
    return q_mes_all

def get_plots(q_mes_all, steps, joints_id, kp):
    # Initialize an empty list to store measured positions
    q_mes_values = []

    # Assume q_mes_all is obtained from some function and represents joint measurements
    q_mes_values.append(q_mes_all)  # Append the measured positions to the list

    # Convert the list of arrays to a 3D numpy array for plotting
    q_mes_values = np.array(q_mes_values)

    # Number of joints to plot (you can change this to however many you need)
    tables = q_mes_values.shape[steps]  # Assuming q_mes_all has shape (num_samples, num_joints)

    # Create subplots for each joint position
    plt.figure(figsize=(15, 5 * tables))  # Adjust figure size as needed

    for joints_id in range(tables):
        plt.subplot(tables, 1, joints_id + 1)  # Create a subplot for each joint
        plt.plot(q_mes_values[:, joints_id], label="Measured Position")
        plt.xlabel('Time step')
        plt.ylabel('Joint Position')
        plt.title(f"Joint {joints_id} Position vs Desired Position (Kp={kp})")
        plt.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
     

# Function to calculate the dominant frequency
def get_dominant_frequency(xf, power):
    # Find the index of the maximum power (peak in the spectrum)
    dominant_freq_idx = np.argmax(power)
    # Get the corresponding frequency
    dominant_frequency = xf[dominant_freq_idx]
    return dominant_frequency


def perform_frequency_analysis(data, dt):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Optional: Plot the spectrum
    plt.figure()
    plt.plot(xf, power)
    plt.title("FFT of the signal")
    plt.xlabel("Frequency in Hz")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return xf, power


# TODO Implement the table in thi function
def calculate_values(Ku, Tu, P=0, PI=0, PD=0, PID=0) -> float:
    """
    Args:


    Return:
        
    """
    table = np.array([[0.5, 0, 0],
             [0.45, 5/6, 0],
             [0.8, 0, 0.125],
             [0.6, 0.5, 0.125]])
    
    mul = np.array([P, PI, PD, PID])

    values = np.matmul(mul, table)

    Kp = Ku*values[0]
    Ki = (Ku*values[0])/(Tu*values[1])
    Kd = (Ku*values[0])*(Tu*values[2])

    return Kp, Ki, Kd



if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=16
    gain_step=0.1
    max_gain=17
    test_duration=2 # in seconds
    q_mes_values = []
    #sustained oscilation Gain = ultiamate gain
    #the period of the frequency at this gain (1/dominant frequency) will be the ultimate period

    # Inside your loop where you analyze the frequency for each gain
    for i in range(int((max_gain - init_gain) / gain_step) + 1):
        # Simulate with the current gain value
        data = simulate_with_given_pid_values(sim_ = sim, kp = init_gain + (i * gain_step), joints_id = joint_id, regulation_displacement=regulation_displacement, episode_duration=test_duration, plot=False)
        # Take out the DC offset of the freuency domain and perform frequency analysis on the data
        data = data - np.mean(data) #takes out the 0 for dominant frequency
        q_mes_values.append(data)
    
    get_plots(q_mes_values, int((max_gain - init_gain) / gain_step), joint_id, init_gain)
    
    joint_data = np.array(data)[:, joint_id]  # Select joint-specific data
    xf, power = perform_frequency_analysis(joint_data, sim.GetTimeStep())


    dom_freq = get_dominant_frequency(xf, power)

    Kp, Ki, Kd = calculate_values(init_gain, 1/dom_freq, PD=1)

    print("kp: ", Kp, "\tKi: ", Ki, "\tKd", Kd)

    # 0: create a while loop to find all the P values with accuracy to 0.01 (use the diference in periods for the 20 seconds and whichever has the smallest difference wins)
    # 1: find Kp for the rest of the joints
    # 2: create visual plot where they are all stable
    # 3: create another plot of all 7 joints and there dominant frequencies, as well as output function for each of them and the Kp values
    # 4: use al Ku values and Tu values to find all the Kp and Kd values
    # 5: use all of these values to show results the PD function working to output all joints
    

    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
