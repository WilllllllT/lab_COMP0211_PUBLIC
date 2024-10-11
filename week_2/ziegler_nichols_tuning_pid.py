import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir)  # Initialize simulation interface

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
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False, kd=0):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joint_id] = kp


    kd_vec = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = kd          # added #

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
        plt.grid(axis="both")
        plt.xlabel('Time step')
        plt.ylabel('Joint Position')
        plt.title(f"Joint {joints_id} Position vs Desired Position (Kp={kp}, Kd={kd})")
        plt.legend()
        plt.show()
    
    return q_mes_all
     

# Function to calculate the dominant frequency
def get_dominant_frequency(xf, power):
    # Find the index of the maximum power (peak in the spectrum)
    dominant_freq_idx = np.argmax(power)
    # Get the corresponding frequency
    dominant_frequency = xf[dominant_freq_idx]
    print(f"Dominant Frequency: {dominant_frequency}")
    return dominant_frequency


def perform_frequency_analysis(data, dt, plot=False):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Null out any 0 value
    power[xf == 0] = 0

    dominant_frequency = get_dominant_frequency(xf, power)

    # Optional: Plot the spectrum
    if plot:
        plt.figure()
        plt.plot(xf, power)
        plt.xlim(0,2)
        plt.title(f"Dominant Frequency of the model, {dominant_frequency} Hz")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return xf, power, dominant_frequency


def calculate_values(Ku, Tu, P=0, PI=0, PD=0, PID=0) -> float:
    """
    Args:
        Ku: the ultimate gain
        Tu: the ultimate period (1/dominant frequency)
        P: True if after P controller
        PI: True if after PI controller
        PD: True if after PD controller
        PID: True if after PID controller

    Return:
        Kp: The proportional constant
        Ki: The integral constant
        Kd: the differential constant
    """
    #ziegler nichols table
    table = np.array([[0.5, 0, 0],
             [0.45, 5/6, 0],
             [0.8, 0, 0.125],
             [0.6, 0.5, 0.125]])
    
    #truth matrix for control type
    mul = np.array([P, PI, PD, PID])

    #outputs the constants for desired controller
    values = np.matmul(mul, table)

    #extractin values from mul
    Kp = Ku*values[0]
    Ki = (Ku*values[0])/(Tu*values[1])
    Kd = (Ku*values[0])*(Tu*values[2])

    return Kp, Ki, Kd

def simulate_over_steps(init_gain, max_gain, step_gain, joint_id, reg_disp, tes_dur, plot):
    """
    quickly passing through multiple values of Kp to find general value for Ku

    Args:
        init_gain: starting value of kp
        max_gain: last value of kp
        step_gain: difference in each step for kp
        joint_id: joint id
        reg_disp: regulation displacement
        tes_dur: the duration of the test
        plot: bool -> true if you want the data plotted

    Return:
        NULL
    """

    for i in range(int((max_gain - init_gain)/step_gain) + 1):
        simulate_with_given_pid_values(sim_ = sim, kp = init_gain + (i * step_gain), joints_id = joint_id, regulation_displacement=reg_disp, episode_duration=tes_dur, plot=plot)

    return

def simulate_over_joints(kp, joint_id, reg_disp, tes_dur, plot, kd):
    """
    quickly passing through multiple values of Kp to find general value for Ku

    Args:
        kp: array of kp to test
        joint_id: number of joints
        reg_disp: regulation displacement
        tes_dur: the duration of the test
        plot: bool -> true if you want the data plotted
        kd: value for kd (default 0)

    Return:
        NULL
    """

    for i in range(joint_id):
        simulate_with_given_pid_values(sim_ = sim, kp = kp[i], joints_id = i, regulation_displacement=reg_disp, episode_duration=tes_dur, plot=plot, kd = kd[i])

    return

def find_ku_and_tu(joint_id, regulation_displacement, test_duration, initial_kp, kp_step, max_kp, plot=False):
    last_dominant_frequency = 0
    last_power_peak = 0
    kp_test = initial_kp
    found = False
    while kp_test <= max_kp and not found:
        q_mes_all = simulate_with_given_pid_values(sim, kp_test, joint_id, regulation_displacement, test_duration, plot)
        joint_data = np.array(q_mes_all)[:, joint_id] 
        joint_data = np.array(joint_data)  # Convert to a numpy array
        joint_data = joint_data - np.mean(joint_data)

        xf, power, dominant_frequency= perform_frequency_analysis(joint_data, sim.GetTimeStep())
        current_power_peak = np.max(power)
        
        if plot:
            plt.figure()
            plt.plot(xf, power)
            plt.title(f"FFT with Kp={kp_test}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        # Check if the dominant frequency stabilizes
        if last_dominant_frequency != 0 and (abs(dominant_frequency - last_dominant_frequency) < 0.3 or current_power_peak < last_power_peak):   # Threshold can be adjusted
            Ku = kp_test
            Tu = 1 / dominant_frequency
            found = True
            print("Ku =", Ku)
            return Ku, Tu, dominant_frequency

        last_dominant_frequency = dominant_frequency
        last_power_peak = current_power_peak
        kp_test += kp_step

    

    if not found:
        raise ValueError("Ku not found within the tested range of Kp")


if __name__ == '__main__':
    # initialise parameters
    joint_id = 1 # Joint ID to tune
    regulation_displacement = 0.1  # Displacement from the initial joint position
    init_gain=1
    gain_step=1
    max_gain=30
    test_duration=20 # in seconds

    #best values for kp without kd
    kp = np.array([16.4, 16, 13.4, 11, 16.6, 16.4, 16.5])

    # Final array of all Kp's and Kd's after ziegler nichols
    kp_final = np.array([13.12, 12.8, 10.72, 8.8, 13.28, 13.12, 13.2])
    kd = np.array([2.52, 2.46, 2.23, 2.0, 2.55, 2.52, 2.54])     

    ### ALL SIM FUNCTIONS USE AND COMMENTED OUT WHEN NOT IN USE ###

    #find_ku_and_tu(joint_id, regulation_displacement, test_duration, initial_kp=init_gain, kp_step=gain_step, max_kp=max_gain, plot=True)

    data = simulate_with_given_pid_values(sim_ = sim, kp = kp[joint_id], joints_id = joint_id, regulation_displacement=regulation_displacement, episode_duration=test_duration, plot=True)
    #simulate_over_steps(init_gain, max_gain, gain_step, joint_id, regulation_displacement, test_duration, True)
    #simulate_over_joints(kp, 7, regulation_displacement, test_duration, True, kd)

    #selecting the joint data that we need
    joint_data = np.array(data)[:, joint_id] 

    # Converting to the frequency domain and showing plot
    xf, power, dom_freq = perform_frequency_analysis(joint_data, sim.GetTimeStep(), True)

    # Calculating the constants via our ziegler nichols table
    Kp, Ki, Kd = calculate_values(init_gain, 1/dom_freq, PD=1)
    print("kp: ", Kp, "\tKi: ", Ki, "\tKd", Kd)

    # Final results
    simulate_with_given_pid_values(sim_ = sim, kp = kp_final[joint_id], joints_id = joint_id, regulation_displacement=regulation_displacement, episode_duration=test_duration, plot=True, kd=kd[joint_id])

