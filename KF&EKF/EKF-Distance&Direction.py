import os
import subprocess
import sys
import time
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt

import rospy as ros
from geometry_msgs.msg import Twist
from rospy import Subscriber
from rospy_tutorials.msg import Floats

BLUE = "#1E64C8"
GREEN = "#71A860"


class KalmanFilterStandingAround:
    """
    This class implements a Kalman filter on a simple example environment, contianing a robot that does not move and a
    single beacon.
    """
    QUEUE_SIZE = 2

    def __init__(self, name: str = "kalman_filter_standing_around"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.beacon_sub_name: str = "/mannequins"
        self.beacon_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.vel_pub_name: str = "/cmd_vel"
        self.vel_pub: Union[None, Subscriber] = None

        # Placeholders for ROS message data
        self.beacons: Union[None, np.ndarray] = None

        # Some housekeeping get the simulation to behave (this might take a few seconds)
        self.__reset_simulation()

        self.fig, _ = plt.subplots()

    def start_ros(self):
        """
        Initialize ros and set all Subscribers and Publishers
        :return:
        """
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.beacon_sub = ros.Subscriber(self.beacon_sub_name, Floats, callback=self.__beacon_ros_sub,
                                         queue_size=self.QUEUE_SIZE)
        self.vel_pub = ros.Publisher(self.vel_pub_name, Twist, queue_size=self.QUEUE_SIZE)

    def run(self):
        """
        Main function of this class. At each timestep, this class sends an input command to the simulated turtlebot3.
        The turtlebot3 will ignore this input command and remain stationary. Then wait a moment for the turtlebot3 to
        finish ignoring the command. Get the sensor measurement, update the current position estimate using the
        Kalman formula's, and plot the new belief over possible states.
        :return:
        """

        # INITIAL GUESS
        mu_t0 = np.array([[0.],
                          [0.]])
        Sigma_t0 = np.array([[3.5 ** 2, 0.000000],
                             [0.000000, 3.5 ** 2]])

        # MOTION COMMANDS THAT WE SEND TO THE TURTLEBOT3 (WHICH THE ROBOT WILL IGNORE)
        U = np.array([[0.5, 0.], [-0., 0.25], [-0.25, 0.25],
                      [-0.5, 0.], [-0.25, -0.25],
                      [0., -0.5], [0.25, -0.25], [0.5, 0.],
                      [0.25, 0.25], [0, 0.25], [-0.5, 0.], [0., 0.]])

        # Set placeholder variables mu_t and Sigma_t to our initial guess, and plot
        mu_t = mu_t0
        Sigma_t = Sigma_t0
        self.__plot_estimated_state_distribution(mu_t, Sigma_t)

        # Loop over input commands that we will send
        for u_index in range(U.shape[0]):
            if ros.is_shutdown():
                break

            # Get the next input command
            u_t = U[u_index][:, None]

            # Send the input command to the robot
            self.__send_input_command(u_t=U[u_index])
            # time.sleep(1.)

            # Get the measurement
            y_t = np.array(list(self.beacons))[:, None]

            # Apply the Kalman filter
            mu_t, Sigma_t = self.kalman_filter(mu_t, Sigma_t, u_t, y_t)

            self.__plot_estimated_state_distribution(mu_t, Sigma_t)

        self.__plot_estimated_state_distribution(mu_t, Sigma_t, display_duration=300)
        ros.signal_shutdown("Done...")
        plt.close(self.fig)

    def kalman_filter(self, mu_old, Sigma_old, u_t, y_t):
        """

        :param mu_old:
        :param Sigma_old:
        :param u_t:
        :param y_t:
        :return:
        """
        """Write down the system for the Kalman filter"""
        if np.all(np.isclose(u_t, 0)):  # no motion
            A = np.array([[1, 0],
                          [0, 1]])
            B = np.array([[0, 0],
                          [0, 0]])
            R = np.array([[0, 0],
                          [0, 0]])
        else:
            A = np.array([[1, 0],
                          [0, 1]])
            B = np.array([[1, 0],
                          [0, 1]])
            R = np.array([[0.03 ** 2, 0.],
                          [0., 0.03 ** 2]])
        mu_t_bar = A @ mu_old + B @ u_t
        Sigma_t_bar = A @ Sigma_old @ A.T + R

        r = np.array([[1.0, 1.5], [1.0, -1.5], [-1.0, -1.5], [-1.0, 1.5]])
        mu = mu_t_bar
        C = np.array(
            [[(r[0, 1] - mu[1, 0]) / ((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2), -(r[0, 0] - mu[0, 0]) / ((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2)],
             [-(r[0, 0] - mu[0, 0]) / np.sqrt((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2), -(r[0, 1] - mu[1, 0]) / np.sqrt((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2)],

             [(r[1, 1] - mu[1, 0]) / ((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2), -(r[1, 0] - mu[0, 0]) / ((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2)],
             [-(r[1, 0] - mu[0, 0]) / np.sqrt((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2), -(r[1, 1] - mu[1, 0]) / np.sqrt((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2)],

             [(r[2, 1] - mu[1, 0]) / ((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2), -(r[2, 0] - mu[0, 0]) / ((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2)],
             [-(r[2, 0] - mu[0, 0]) / np.sqrt((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2), -(r[2, 1] - mu[1, 0]) / np.sqrt((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2)],

             [(r[3, 1] - mu[1, 0]) / ((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2), -(r[3, 0] - mu[0, 0]) / ((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2)],
             [-(r[3, 0] - mu[0, 0]) / np.sqrt((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2), -(r[3, 1] - mu[1, 0]) / np.sqrt((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2)]])

        G = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, np.sqrt((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2), 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, np.sqrt((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2), 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, np.sqrt((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2), 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, np.sqrt((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2)]])
        Q = G @ (np.identity(8) * 0.25 ** 2) @ G.T

        y_bar = np.array([[np.arctan2(r[0, 1] - mu[1, 0], r[0, 0] - mu[0, 0])],
                          [np.sqrt((r[0, 0] - mu[0, 0]) ** 2 + (r[0, 1] - mu[1, 0]) ** 2)],

                          [np.arctan2(r[1, 1] - mu[1, 0], r[1, 0] - mu[0, 0])],
                          [np.sqrt((r[1, 0] - mu[0, 0]) ** 2 + (r[1, 1] - mu[1, 0]) ** 2)],

                          [np.arctan2(r[2, 1] - mu[1, 0], r[2, 0] - mu[0, 0])],
                          [np.sqrt((r[2, 0] - mu[0, 0]) ** 2 + (r[2, 1] - mu[1, 0]) ** 2)],

                          [np.arctan2(r[3, 1] - mu[1, 0], r[3, 0] - mu[0, 0])],
                          [np.sqrt((r[3, 0] - mu[0, 0]) ** 2 + (r[3, 1] - mu[1, 0]) ** 2)],
                          ])

        # This line is about the modulo 2pi issue on the direction coordinate
        y_tilde = y_t - y_bar
        y_tilde[[0, 2, 4, 6]] = (y_tilde[[0, 2, 4, 6]] + np.pi) % (2 * np.pi) - np.pi # only on the indices that measure angle

        """Now we implement the Kalman filter formulas"""
        # Calculate Kalman gain
        K = Sigma_t_bar @ C.T @ np.linalg.inv(C @ Sigma_t_bar @ C.T + Q)

        # Update mu and sigma based on new information
        mu_t = mu_t_bar + K @ y_tilde
        sigma_t = (np.identity(2) - K @ C) @ Sigma_t_bar

        # return the updated state estimate
        return mu_t, sigma_t

    def __send_input_command(self, u_t):
        """
        This method attempts to send an input command to the simulated Turtlebot3. For some reason the Turtlebot3 is
        ignoring all our messages, but we will deal with that later.
        :param u_t:
        :return:
        """
        if True:
            msg = Twist()
            msg.linear.x = u_t[0]
            msg.linear.y = u_t[1]
            self.vel_pub.publish(msg)
        time.sleep(1.)

    def __plot_estimated_state_distribution(self, mu, Sigma, display_duration=0.1):
        """
        Plot a multivatiate gaussian with mean mu and covariance matrix Sigma in a heatmap and display the heatmap for
        a minimal duration of display_duration.
        :param mu:
        :param Sigma:
        :param display_duration:
        :return:
        """

        # Create an (evenly spaced) meshgrid over x1 and x2 axes
        x1 = np.linspace(-1.25, 1.25, 125)
        x2 = np.linspace(-1.75, 1.75, 175)
        xx1, xx2 = np.meshgrid(x1, x2)
        X = np.stack((xx1, xx2), axis=-1)

        # 2D multivariate Gaussian probability distribution over possible locations
        D = 2
        nominator = np.exp(
            -0.5 *
            (X[..., None] - mu[None, None, :]).transpose(0, 1, 3, 2)
            @ np.linalg.inv(Sigma)[None, None, :, :]
            @ (X[..., None] - mu[None, None, :])).squeeze((2, 3))
        denominator = (2 * np.pi) ** (D / 2) * np.linalg.det(Sigma) ** (1 / 2)
        P = nominator / denominator

        # Make a plot
        heatmap = self.fig.axes[0].imshow(P[::-1], extent=[-1.25, 1.25, -1.75, 1.75], cmap="plasma")
        self.fig.axes[0].scatter(1., 1.5, c=GREEN, marker='o')
        self.fig.axes[0].scatter(1., -1.5, c=GREEN, marker='o')
        self.fig.axes[0].scatter(-1., -1.5, c=GREEN, marker='o')
        self.fig.axes[0].scatter(-1., 1.5, c=GREEN, marker='o')
        colorbar = plt.colorbar(heatmap, ax=self.fig.axes[0])
        self.__display_opencv_image(duration=display_duration)
        colorbar.remove()

    def __display_opencv_image(self, duration=0.1):
        self.fig.canvas.draw()
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("kalman.jpg", img)
        cv2.imshow("Exercise 5", img)

        t0 = time.time()
        cv2.waitKey(50)
        while time.time() - t0 < duration:
            try:
                cv2.getWindowProperty("Exercise 5", 0)
                cv2.waitKey(50)
            except cv2.error:
                break

    def __beacon_ros_sub(self, msg):
        self.beacons = msg.data

    def __reset_simulation(self):
        """
        This method resets the gazebo simulator and the  PID controller that we use to fake noiseless
        omni-drive. You can ignore this function
        :return:
        """
        os.system("rostopic pub /reset std_msgs/Bool true -1")
        os.system("rosservice call /gazebo/reset_world")


if __name__ == '__main__':
    kf = KalmanFilterStandingAround()
    kf.start_ros()
    kf.run()
