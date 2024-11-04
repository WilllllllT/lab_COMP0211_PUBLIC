#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

#initial conditions for the filter
class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements)
        self.W_range = 0.5 ** 2
        self.W_bearing = (np.pi * 0.5 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2

#setting the land marks for the map
class Map(object):
    def __init__(self):
        # # Define the radii and number of landmarks for each ring
        # ring_radii = [10, 2]  # Add more radii as needed for more rings
        # num_landmarks_per_ring = [12, 8]  # Number of landmarks for each ring

        # # Initialize list to hold all landmark coordinates
        # landmarks = []

        # # Generate landmarks for each ring
        # for radius, num_landmarks in zip(ring_radii, num_landmarks_per_ring):
        #     # Create angles evenly spaced around the circle for the current ring
        #     angles = np.linspace(0, 2 * np.pi, num_landmarks, endpoint=False)
            
        #     # Generate (x, y) coordinates for each landmark on the ring
        #     ring_landmarks = [
        #         [radius * np.cos(angle), radius * np.sin(angle)] for angle in angles
        #     ]
            
        #     # Add current ring's landmarks to the full list
        #     landmarks.extend(ring_landmarks)

        # # Optionally add a central landmark at [0,0]
        # landmarks.append([0, 0])

        # # Convert the landmarks list to a NumPy array for consistency
        # self.landmarks = np.array(landmarks)


        #####################circle pattern################
        # # Define radii for the outer and inner circles
        # outer_radius = 35  # Radius of the outer circle
        # inner_radius = 15  # Radius of the inner circle

        # # Define the number of landmarks for each circle
        # num_landmarks_outer = 20  # Number of landmarks on the outer circle
        # num_landmarks_inner = 10  # Number of landmarks on the inner circle

        # # Create outer circle landmarks
        # angles_outer = np.linspace(0, 2 * np.pi, num_landmarks_outer, endpoint=False)
        # outer_circle_landmarks = [
        #     [outer_radius * np.cos(angle), outer_radius * np.sin(angle)] for angle in angles_outer
        # ]

        # # Create inner circle landmarks
        # angles_inner = np.linspace(0, 2 * np.pi, num_landmarks_inner, endpoint=False)
        # inner_circle_landmarks = [
        #     [inner_radius * np.cos(angle), inner_radius * np.sin(angle)] for angle in angles_inner
        # ]



        # # Combine the landmarks from both circles
        # self.landmarks = np.array(outer_circle_landmarks + inner_circle_landmarks + [[0,0]])
        ##################################################

        #################grid pattern#################
        # Generate a grid of landmark coordinates from -30 to 30 with a step of 5
        # Define denser range for border points (more frequent points along the edges)
        
        # # Define the range and step size for the border (points along the edges only)
        # x_border = np.concatenate((np.arange(-40, 41, 20), np.array([-40, 40])))  # x coordinates at edges and corners
        # y_border = np.concatenate((np.arange(-40, 41, 20), np.array([-40, 40])))

        # # Define the inner grid with a smaller range
        # x_inner = np.arange(-10, 11, 20)  # Inner grid range
        # y_inner = np.arange(-10, 11, 20)

        # # Create border points using only edge points of the defined border range
        # xx_border = np.concatenate([np.full(len(y_border), -40), np.full(len(y_border), 40), x_border, x_border])
        # yy_border = np.concatenate([y_border, y_border, np.full(len(x_border), -40), np.full(len(x_border), 40)])

        # # Stack border coordinates
        # landmarks_border = np.column_stack((xx_border, yy_border))

        # # Create the inner grid points
        # xx_inner, yy_inner = np.meshgrid(x_inner, y_inner)
        # landmarks_inner = np.column_stack([xx_inner.ravel(), yy_inner.ravel()])

        # # Combine border and inner grid points
        # landmarks = np.vstack((landmarks_border, landmarks_inner))
        # landmarks = np.unique(landmarks, axis=0)



        # # Set the landmarks as an attribute
        # self.landmarks = landmarks
        ##########################################

        ################Grid Pattern################
        # Define the range for x and y coordinates
        x_values = range(-10, 10, 2)  
        y_values = range(-10, 10, 2)  

        # Create a list to store the grid coordinates
        grid_pattern = [[x, y] for x in x_values for y in y_values]
        
        self.landmarks = np.array(grid_pattern)
        #########################################

        # self.landmarks = np.array([
        #     [5, 10],
        #     [15, 5],
        #     [10, 15]
        # ])


class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    # This nethod MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):
        # error check the dimensions
    #     print("C: ", C.shape)
    #     print("W: ", W.shape)
    #     print("Sigma_pred: ", self._Sigma_pred.shape)
    #     print("nu: ", nu.shape)

        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu

        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred

    

    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:

            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))
        

    #TODO: implement the bearing and range update landmark method
    def update_from_landmark_observations(self, y_range ,y_bearing):
        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred_r = []
        y_pred_b = []
        C_r = []
        C_b = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            theta_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            
            #fix dy and dx to show the range by taking the square root of the sum of the squares
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)

            #wrap
            theta_pred = np.arctan2(np.sin(theta_pred), np.cos(theta_pred))

            #wrap the bearing prediction again to keep results consistent
            y_pred_r.append(range_pred)
            y_pred_b.append(theta_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])

            # Jacobian of the measurement model using equation 9
            C_bearing = np.array([
                dy_pred / (dx_pred**2 + dy_pred**2),
                -dx_pred / (dx_pred**2 + dy_pred**2),
                -1
            ])

            C_r.append(C_range)
            C_b.append(C_bearing)
        
        # Convert lists to arrays
        C_r = np.array(C_r)
        C_b = np.array(C_b)
        C = np.vstack((C_r, C_b)) #matrix of size 2n x 3

        y_pred_r = np.array(y_pred_r) # size n x 1
        y_pred_b = np.array(y_pred_b)

        # Innovation. Look new information! (geddit?)
        nu_r = y_range - y_pred_r
        nu_b = y_bearing - y_pred_b 

        #wrapping the bearing error
        nu_b = np.arctan2(np.sin(nu_b), np.cos(nu_b))
        nu = np.hstack((nu_r, nu_b)) #matrix of size 2n x 1
        

        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks_r = self._config.W_range * np.eye(len(self._map.landmarks))
        W_landmarks_b = self._config.W_bearing * np.eye(len(self._map.landmarks))

        # combine the two covariance matrices to create a 6x6 matrix
        W = np.block([
            [W_landmarks_r, np.zeros_like(W_landmarks_r)],
            [np.zeros_like(W_landmarks_b), W_landmarks_b]
        ]) #matrix of size 2n x 2n

        self._do_kf_update(nu, C, W)

        print("size of nu: ", nu.shape)
        print("size of C: ", C.shape)
        print("size of W: ", W.shape)


        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))

