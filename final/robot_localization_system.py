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
        self.landmarks = np.array([
            # [5, 5],
            # [5, 10],
            # [5, 15],
            # [10, 5],
            # [10, 10],
            # [10, 15],
            # [15, 5],
            # [15, 10],
            # [15, 15]

            # confirmed works for range and bearing
            [-15, -15], [-15, -10], [-15, -5], [-15, 0], [-15, 5], [-15, 10], [-15, 15],
            [-10, -15], [-10, -10], [-10, -5], [-10, 0], [-10, 5], [-10, 10], [-10, 15],
            [-5, -15], [-5, -10], [-5, -5], [-5, 0], [-5, 5], [-5, 10], [-5, 15],
            [0, -15], [0, -10], [0, -5], [0, 0], [0, 5], [0, 10], [0, 15],
            [5, -15], [5, -10], [5, -5], [5, 0], [5, 5], [5, 10], [5, 15],
            [10, -15], [10, -10], [10, -5], [10, 0], [10, 5], [10, 10], [10, 15],
            [15, -15], [15, -10], [15, -5], [15, 0], [15, 5], [15, 10], [15, 15]
            
            
            # [-10, -10], [-10, -5], [-10, 0], [-10, 5], [-10, 10], [-10, 15], [-10, 20], [-10, 25], [-10, 30],
            # [-5, -10], [-5, -5], [-5, 0], [-5, 5], [-5, 10], [-5, 15], [-5, 20], [-5, 25], [-5, 30],
            # [0, -10], [0, -5], [0, 0], [0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30],
            # [5, -10], [5, -5], [5, 0], [5, 5], [5, 10], [5, 15], [5, 20], [5, 25], [5, 30],
            # [10, -10], [10, -5], [10, 0], [10, 5], [10, 10], [10, 15], [10, 20], [10, 25], [10, 30],
            # [15, -10], [15, -5], [15, 0], [15, 5], [15, 10], [15, 15], [15, 20], [15, 25], [15, 30],
            # [20, -10], [20, -5], [20, 0], [20, 5], [20, 10], [20, 15], [20, 20], [20, 25], [20, 30],
            # [25, -10], [25, -5], [25, 0], [25, 5], [25, 10], [25, 15], [25, 20], [25, 25], [25, 30],
            # [30, -10], [30, -5], [30, 0], [30, 5], [30, 10], [30, 15], [30, 20], [30, 25], [30, 30]

            #confirmed works entierly for just range
            # [-20, -10], [-20, -5], [-20, 0], [-20, 5], [-20, 10], [-20, 15], [-20, 20], [-20, 25], [-20, 30],
            # [-15, -10], [-15, -5], [-15, 0], [-15, 5], [-15, 10], [-15, 15], [-15, 20], [-15, 25], [-15, 30],
            # [-10, -10], [-10, -5], [-10, 0], [-10, 5], [-10, 10], [-10, 15], [-10, 20], [-10, 25], [-10, 30],
            # [-5, -10], [-5, -5], [-5, 0], [-5, 5], [-5, 10], [-5, 15], [-5, 20], [-5, 25], [-5, 30],
            # [0, -10], [0, -5], [0, 0], [0, 5], [0, 10], [0, 15], [0, 20], [0, 25], [0, 30],
            # [5, -10], [5, -5], [5, 0], [5, 5], [5, 10], [5, 15], [5, 20], [5, 25], [5, 30],
            # [10, -10], [10, -5], [10, 0], [10, 5], [10, 10], [10, 15], [10, 20], [10, 25], [10, 30],
            # [15, -10], [15, -5], [15, 0], [15, 5], [15, 10], [15, 15], [15, 20], [15, 25], [15, 30],
            # [20, -10], [20, -5], [20, 0], [20, 5], [20, 10], [20, 15], [20, 20], [20, 25], [20, 30],
            # [25, -10], [25, -5], [25, 0], [25, 5], [25, 10], [25, 15], [25, 20], [25, 25], [25, 30],
            # [30, -10], [30, -5], [30, 0], [30, 5], [30, 10], [30, 15], [30, 20], [30, 25], [30, 30]




            # [5, 10],
            # [15, 5],
            # [10, 15]
        ])


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
        count = 0
        if count == 0:
            print("C: ", C.shape)
            print("W: ", W.shape)
            print("Sigma_pred: ", self._Sigma_pred.shape)
            print("nu: ", nu.shape)
            count += 1


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
            
            #calculate the predictions for bearing and range
            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)

            #wrap the bearing prediction again to keep results consistent
            y_pred_r.append(range_pred)
            y_pred_b.append(bearing_pred)

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
        C = np.vstack((C_r, C_b))

        y_pred_r = np.array(y_pred_r)
        y_pred_b = np.array(y_pred_b)

        # Innovation. Look new information! (geddit?)
        nu_r = y_range - y_pred_r
        nu_b = y_bearing - y_pred_b

        #wrapping the bearing error
        nu_b = np.arctan2(np.sin(nu_b), np.cos(nu_b))
        nu = np.hstack((nu_r, nu_b))
        

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
        ])

        self._do_kf_update(nu, C, W)


        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))

