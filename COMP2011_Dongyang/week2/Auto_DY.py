import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin

# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# Single joint tuning
# Episode duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, kd, joints_id, regulation_displacement=0.1, episode_duration=10, plot=False):
    
    # Reset the simulator each time a new test starts
    sim_.ResetPose()
    
    # Update the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd_vec = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = kd

    # IMPORTANT: To ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())
    
    time_step = sim_.GetTimeStep()
    q_des[joints_id] += regulation_displacement 
    steps = int(episode_duration / time_step)
    
    current_time = 0
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all, time_all, regressor_all = [], [], [], [], [], []

    steps = int(episode_duration/time_step)
    # Testing loop
    for i in range(steps):
        # Measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting
        q_mes_all.append(q_mes[joints_id])
        qd_mes_all.append(qd_mes[joints_id])
        q_d_all.append(q_des[joints_id])
        qd_d_all.append(qd_des[joints_id])
        time_all.append(i * time_step)

        current_time += time_step
        if(steps%1000 == 0):
            print("Current time in seconds", current_time, "\tfor: ", kp)

    # Make the plot for the current joint if required
    if plot:
        plt.figure()
        plt.plot(time_all, q_mes_all, label='Measured Joint Angles')
        plt.plot(time_all, q_d_all, 'r--', label='Desired Joint Angle')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Joint Angle (radians)")
        plt.title("Joint Angle Response Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return q_mes_all

def perform_frequency_analysis(data, dt, plot=False):
    data = np.array(data)  # Convert to a numpy array
    data = data - np.mean(data)
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])
    
    # Find the frequency point with the maximum amplitude
    idx_max = np.argmax(power)
    dominant_frequency = xf[idx_max]

    # Optional: Plot the spectrum
    if plot:
        plt.figure()
        plt.plot(xf, power)
        plt.xlim(0,2)
        plt.title("FFT of the signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    print(f"frequency = {dominant_frequency}")

    return xf, power, dominant_frequency

def find_ku_and_tu(sim, joint_id, regulation_displacement, test_duration, initial_kp=15, kp_step=1, max_kp=18, plot=False, stability_threshold=3):
    last_power_peak = 0
    closest_ku = None
    closest_tu = None
    min_power_diff = float('inf')  # To track the closest power difference
    stable_count = 0  # Counter for consecutive stable oscillations

    kd_test = 0  # Assuming kd is zero for this process

    # Iterate over the Kp values in a for loop
    for kp_test in range(int((max_kp - initial_kp) / kp_step) + 1):
        q_mes_all = simulate_with_given_pid_values(sim, initial_kp + (kp_test * kp_step), kd_test, joint_id, regulation_displacement, test_duration, plot)
        xf, power, dominant_frequency = perform_frequency_analysis(q_mes_all, sim.GetTimeStep(), plot)
        current_power_peak = np.max(power)

        if plot:
            plt.figure()
            plt.plot(xf, power)
            plt.title(f"FFT with Kp={initial_kp + (kp_test * kp_step)}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        # Check if the oscillation is stable based on power differences
        if last_power_peak != 0:
            power_diff = abs(current_power_peak - last_power_peak)
            
            # Check if the current power peak is stable
            if power_diff < 0.05:  # Assuming a threshold for power stability
                stable_count += 1  # Increment stable count
                
                # If we found a closer Ku, update closest_ku and closest_tu
                if power_diff < min_power_diff:
                    closest_ku = initial_kp + (kp_test * kp_step)
                    closest_tu = 1 / dominant_frequency
                    min_power_diff = power_diff
            else:
                stable_count = 0  # Reset stable count if not stable
        else:
            stable_count = 0  # Reset on first iteration

        # Update for the next iteration
        last_power_peak = current_power_peak

    # If we found a closest Ku and it has been stable for enough iterations, return it
    if closest_ku is not None and stable_count >= stability_threshold:
        return closest_ku, closest_tu, dominant_frequency

    # If no suitable Ku was found, raise an error
    raise ValueError("Ku not found within the tested range of Kp or oscillations were not stable.")



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

def main():
    joint_id = 1 # The joint to be tuned
    regulation_displacement = 1
    test_duration = 20
    plot_results = True
    initial_kp = 10
    kp_step = 0.1
    max_kp = 10

    # Find Ku and Tu
    try:
        Ku, Tu, dominant_frequency = find_ku_and_tu(sim, joint_id, regulation_displacement, test_duration, initial_kp, kp_step, max_kp, plot=plot_results, stability_threshold=3)
        print(f"Found Ku: {Ku}, Tu: {Tu}, with dominant frequency: {dominant_frequency} Hz")

        # Calculate Kp and Kd based on Ziegler-Nichols formula for a PID controller
        Kp, Ki, Kd = calculate_values(Ku, Tu, PD=True)

        # Re-run the simulation with the calculated Kp and Kd
        q_mes_all = simulate_with_given_pid_values(sim, Kp, Kd, joint_id, regulation_displacement, test_duration, plot=True)
        perform_frequency_analysis(q_mes_all, sim.GetTimeStep())
        print(f"Calculated Kp: {Kp}, Kd: {Kd}")
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
