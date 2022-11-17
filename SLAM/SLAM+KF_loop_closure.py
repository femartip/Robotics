import atexit
import multiprocessing
import os
import random
import subprocess
import time
from tkinter import TclError
from typing import Union, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from numpy import cos, sin
from numpy.linalg import linalg

import rospy as ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rospy import Subscriber
from rospy_tutorials.msg import Floats
from tf.transformations import euler_from_quaternion

BLUE = "#1E64C8"
GREEN = "#71A860"


class SLAMLoopClosure:
    """
    This class implements SLAM with a Kalman filter. The requirement for this is that the robot
    never confuses two beacons.
    """
    QUEUE_SIZE = 2

    def __init__(self, name: str = "slam_standard_kalman_filter"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.beacon_sub_name: str = "/mannequins"
        self.beacon_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.vel_pub_name: str = "/closedloop_cmd"
        self.vel_pub: Union[None, Subscriber] = None

        # Placeholders for ROS message data
        self.beacons: Union[None, np.ndarray] = None

        # Some housekeeping for the simulation and visualization
        self.__reset_simulation()
        self.fig = Figure().start()

    def start_ros(self):
        """
        Initialize ros and set all Subscribers and Publishers
        """
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.beacon_sub = ros.Subscriber(self.beacon_sub_name, Floats,
                                         callback=self.__beacon_ros_sub,
                                         queue_size=self.QUEUE_SIZE)
        self.vel_pub = ros.Publisher(self.vel_pub_name, Twist, queue_size=self.QUEUE_SIZE)
        time.sleep(1)

    def run(self):
        # MOTION COMMANDS THAT WE SEND TO THE TURTLEBOT3
        trn = 0.1 * np.pi
        U = np.array([[trn, 0.1], [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1], [0., 0.2],
                      [trn, 0.1], [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1], [trn, 0.1],
                      [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1], [0., 0.2], [trn, 0.1],
                      [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1], [-trn, 0.1], [-trn, 0.1],
                      [-trn, 0.2], [-trn, 0.2], [-trn, 0.1], [0., 0.2], [-trn, 0.1], [-trn, 0.1],
                      [-trn, 0.2], [-trn, 0.2], [-trn, 0.1], [-trn, 0.1], [-trn, 0.1], [-trn, 0.2],
                      [-trn, 0.2], [-trn, 0.1], [0., 0.2], [-trn, 0.1], [-trn, 0.1], [-trn, 0.2],
                      [-trn, 0.2], [-trn, 0.1], [0., 0.0], [0., 0.0], [0., 0.0], [0., 0.0],
                      [0., 0.0]])
        dt = 1  # seconds

        # INITIAL GUESS. We start with a high degree of certainty that the robot starts at (0, 0)
        # and with high UNcertainty about the position of the beacons. Because we now have an
        # 11-dimensional state vector, we stop writing matrices in full and start using some proper
        # numpy logic. You can still view the vectors and matrices in a readable form in the
        # debugger after they have been created.
        # Let's assume that the beacons are quite far away at the beginning so that we do not make
        # any overconfident predictions (due to relative noise) the first time we see a beacon
        mu_t0 = np.concatenate((np.zeros((3, 1)), np.full((8, 1), 10)), axis=0)
        Sigma_t0 = np.identity(11) * 100 ** 2
        Sigma_t0[:3, :3] = np.identity(3) * 1e-6 # High certainty over initial pose robot

        # Set placeholder variables mu_t and Sigma_t to our initial guess, and plot
        mu_t, Sigma_t = mu_t0, Sigma_t0
        self.fig.plot_state_estimate(mu_t, Sigma_t)

        for u_index in range(U.shape[0]):
            if ros.is_shutdown():
                break

            # Get the next input command
            u_t = U[u_index][:, None]

            # Send the input command to the robot
            self.__send_input_command(u_t=U[u_index], duration=dt)

            # Get measurement
            y_t = self.beacons

            # Apply the Kalman filter
            mu_t_bar, Sigma_t_bar = self.kalman_motion_update(mu_t, Sigma_t, u_t, dt)
            mu_t, Sigma_t = self.kalman_sensor_update(mu_t_bar, Sigma_t_bar, u_t, y_t, dt)

            self.fig.plot_state_estimate(mu_t, Sigma_t)

        self.fig.stop()
        self.__send_input_command([0, 0])

    def kalman_motion_update(self, mu_old, Sigma_old, u_t, dt):

        # For readability, we make copies of mu_old and u_t
        mu, u = mu_old[:, 0], u_t[:, 0]

        # For readability, we compute the repeating terms in the motion model here
        dx = u[1] * dt if u[0] == 0. else u[1] * sin(u[0] * dt) / u[0]
        dy = 0 if u[0] == 0. else u[1] * (1 - cos(u[0] * dt)) / u[0]

        # Compute mu_bar, by filling in mu_old in the motion model and assuming zero noise
        mu_bar = mu_old  # beacons do not move
        mu_bar[:3] = np.array([[mu[0] + cos(mu[2]) * dx - sin(mu[2]) * dy],
                               [mu[1] + sin(mu[2]) * dx + cos(mu[2]) * dy],
                               [mu[2] + u[0]]])

        # A is the Jacobian of the motion model with respect to x
        A = np.identity(11)  # beacons do not move
        A[:3, :3] = np.array([[1, 0, sin(mu[2]) * dx + cos(mu[2]) * dy],
                              [0, 1, cos(mu[2]) * dx + sin(mu[2]) * dy],
                              [0, 0, 1]])
        # Noise depends on motion command
        stdev = 0. if u[0] == 0 and u[1] == 0 else 0.05 * dt
        R = np.zeros((11, 11))  # No motion noise on beacons
        R[:3, :3] = np.identity(3) * stdev ** 2  # No need to linearize here

        Sigma_bar = A @ Sigma_old @ A.T + R

        return mu_bar, Sigma_bar

    def kalman_sensor_update(self, mu_bar, Sigma_bar, u_t, y_t, dt):
        """
        Kalman sensor update step where beacons are not always visible

        :param mu_bar: should contain position estimate of the robot and of the four beacons
                       (vector of length 11)
        :param Sigma_bar: should contain covariance matrix (matrix 11x11)
        :param u_t: contains the input commant at time t (vector of length 2)
        :param y_t: contains the measurement data (vector of length 8)
        :param dt:
        :return:
        """

        if np.all(np.isnan(y_t)):
            # If no measurements were received (no beacons visible) we cannot update our estimate,
            # so simply return the best estimate we have so far: mu_bar and Sigma_bar
            return mu_bar, Sigma_bar
        else:
            # If we did get a measurement, construct the C-matrix like we did before ...
            C = np.concatenate((np.concatenate((-np.identity(2),) * 4),
                                np.zeros((8, 1)),
                                np.identity(8)), axis=1)
            # ... but set all rows corresponding to unseen beacons to 0 (i.e. no relation between
            # beacon and measurement)
            C[np.isnan(y_t.squeeze())] = 0.

            # Second modification: say that noise is proportional to (estimated) distance
            distance_per_beacon = np.sqrt(
                (mu_bar[None, 0] - mu_bar[3::2]) ** 2 + (mu_bar[None, 1] - mu_bar[4::2]) ** 2)
            # Each beacon yields two measurements, so repeat distance_per_beacon to get 8x1
            # distances
            distance_per_measurement = np.repeat(distance_per_beacon, 2)
            # Now compute Q based on the sensor model
            Q = np.identity(8) * distance_per_measurement * 0.05 ** 2
            # (you could set rows corresponding to unseen beacons to 0 in the Q-matrix as well, but
            # you do not need to. We end up multiplying these values with the zeros in the C-matrix
            # during the Kalman formulas)

            # Before doing the Kalman formulas, Python cannot do "nan * 0 = 0", so replace all nans
            # in y_t with an arbitrary number (we don't use these numbers in the end due to the
            # zeros in the C-matrix)
            y_t[np.isnan(y_t.squeeze())] = 42

            # Now the Kalman formulas
            K = Sigma_bar @ C.T @ np.linalg.inv(C @ Sigma_bar @ C.T + Q)
            mu = mu_bar + K @ (y_t - C @ mu_bar)
            Sigma = (np.identity(11) - K @ C) @ Sigma_bar

            return mu, Sigma

    def __send_input_command(self, u_t, duration=1.):
        """
        Send a velocity command to the closed-loop controller
        :param u_t:
        :param duration:
        :return:
        """
        msg = Twist()
        msg.angular.z = u_t[0]
        msg.linear.x = u_t[1]
        self.vel_pub.publish(msg)
        time.sleep(duration)

    def __beacon_ros_sub(self, msg):
        """
        Process new beacon measurement
        :return:
        """
        self.beacons = np.array(list(msg.data))[:, None]

    def __reset_simulation(self):
        """
        Reset simulation and wait a few seconds for Gazebo to catch up
        :return:
        """

        os.system("rostopic pub /reset std_msgs/Bool true -1")
        os.system("rosservice call /gazebo/reset_world")


class Figure:
    """
    Class for plotting uncertainty over robot position. You do not need to edit anything in this
    class. Because we are working with the real robot model in Gazebo, timing is important and
    plotting would be too computationally intensive to do in the main loop. We have made a class
    that does all the heavy computations for rendering in a parallel process.
    """

    def __init__(self, title="Local SLAM"):
        self.title = title

        # Define grid over X and theta
        self.X = np.stack(np.meshgrid(np.linspace(-1.25, 1.25, 125),
                                      np.linspace(-1.75, 1.75, 175)), axis=-1)
        self.theta = np.linspace(-np.pi, np.pi, 360)[:, None]

        # A queue is used to communicate between processes. Python is not well optimized for
        # multithreading. Upon initialisation, multiprocessing makes a copy of all visible
        # variables and starts a new process in the background. This copy is made entirely
        # by-value and not by-reference. So if you want to send data between processes, you need
        # to do so explicitly with a socket connection, such as multiprocessing.Pipe or
        # multiprocessing.Queue
        self.queue = multiprocessing.Queue()
        self.process: Union[None, multiprocessing.Process] = None

    def start(self):
        # start a parallel process
        self.process = multiprocessing.Process(target=self.__run, args=(self.queue,))
        self.process.start()

        # atexit registers a function that will be called once the program finishes or is
        # is interruped. In this case, we close all the opencv windows.
        atexit.register(cv2.destroyAllWindows)
        return self

    def stop(self):
        # Send a signal to the parallel process to let it know that we want to stop.
        self.queue.put(False)

        # Wait until the parallel process finishes
        self.process.join()

    def plot_state_estimate(self, mu, Sigma):
        # Send the state estimate to the parallel program
        self.queue.put((mu, Sigma))

    def __run(self, queue: multiprocessing.Queue):
        """
        This is the main loop of the parallel program. So it runs independently from our main
        program; all communication goes through the queue.
        :param queue:
        :return:
        """

        for f in os.listdir("./res"):
            os.remove(f"./res/{f}")

        # Create figure
        np.set_printoptions(precision=2)
        fig = plt.figure(figsize=(4.20, 8.00))
        ax0: Axes = plt.axes([0.1, 0.48, 0.8, 0.42])
        ax1: Axes = plt.axes([0.1, 0.1, 0.8, 0.28], polar=True)

        img = None
        while True:
            # Wait for more data
            if queue.empty():
                time.sleep(0.1)
            msg = queue.get()

            # If the received message is a simple boolean False, this is the exit signal.
            # Display the last figure for a long time, then exit the main loop once the user closes
            # the figure.
            if not msg:
                self.__display_opencv_image(fig, duration=300)
                break

            # Message contains state estimat
            mu, Sigma = msg

            # Plot the heatmap of estimated robot location
            ax0.clear()
            ax0.imshow(self.__gaussian(self.X, mu[:2], Sigma[:2, :2])[::-1],
                       extent=[-1.25, 1.25, -1.75, 1.75], cmap="plasma")
            self.__uncertainty_ellipse(ax0, mu[3:], Sigma[3:, 3:])

            # Plot the polar plot of estimated robot orientation
            ax1.clear()
            ax1.plot(self.theta + mu[2:3],
                     self.__gaussian(self.theta + mu[2:3], mu[2:3], Sigma[2:3, 2:3]))

            # Display the figure in a window. If the user closes the window, this function returns
            # False and we exit the main loop.
            if not self.__display_opencv_image(fig):
                break

        # Close any windows that were still open
        cv2.destroyAllWindows()

    def __uncertainty_ellipse(self, ax, mu, Sigma):
        """
        Draw uncertainty ellipses around the estimated locations of beacons
        :param ax:
        :param mu:
        :param Sigma:
        :return:
        """
        # Do not rescale image if (part of) an ellipse goes beyond the border of the heatmap.
        ax.autoscale(False)
        for ii in range(0, mu.shape[0], 2):  # Loop over beacons
            mu_ii = mu[ii: ii + 2]
            Sigma_ii = Sigma[ii: ii + 2, ii:ii + 2]

            # Draw an ellipse 3 standard deviations around the estimated beacon position
            w, v = np.linalg.eigh(Sigma_ii)
            el = Ellipse(xy=mu_ii, width=3 * np.sqrt(w[0]), height=3 * np.sqrt(w[1]),
                         angle=np.arctan2(v[1, 0], v[0, 0]), color=GREEN, linewidth=2, fill=False)
            ax.add_patch(el)

    def __gaussian(self, X, mu, Sigma):
        """
        Vectorized implementation of the gaussian function.
        :param X:
        :param mu:
        :param Sigma:
        :return:
        """
        shape = X.shape
        x = X.reshape((np.prod(shape[:-1]), shape[-1]))

        nominator = np.exp(-0.5 * (x[..., None] - mu[None, ...]).transpose(0, -1, -2)
                           @ linalg.inv(Sigma)[None, ...]
                           @ (x[..., None] - mu[None, ...])).squeeze((-1, -2))
        denominator = (2 * np.pi) ** (mu.shape[0] / 2) * linalg.det(Sigma) ** .5
        return np.reshape(nominator / denominator, shape[:-1])

    def __display_opencv_image(self, figure, duration=0.1):
        """
        Display the live plot with opencv
        :param figure:
        :param duration:
        :return:
        """
        figure.canvas.draw()
        img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.title, img)
        cv2.imwrite(f"./res/{len(os.listdir('./res'))}.jpg", img)
        t0 = time.time()
        cv2.waitKey(50)
        while True:
            try:
                cv2.getWindowProperty(self.title, 0)
                if time.time() - t0 > duration:
                    return True
                cv2.waitKey(50)
            except cv2.error:
                return False


if __name__ == '__main__':
    slam = SLAMLoopClosure()
    slam.start_ros()
    slam.run()
