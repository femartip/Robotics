import atexit
import multiprocessing
import os
import random
import time
from typing import Union, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from numpy import sin, cos

import rospy as ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rospy import Subscriber
from rospy_tutorials.msg import Floats
from tf.transformations import euler_from_quaternion

BLUE = "#1E64C8"
GREEN = "#71A860"


class SLAMParticleFilter:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 50

    def __init__(self, name: str = "slam_particle_filter"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.beacon_sub_name: str = "/mannequins"
        self.beacon_sub: Union[None, Subscriber] = None
        self.odom_sub_name: str = "/odom"
        self.odom_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.vel_pub_name: str = "/cmd_vel"
        self.vel_pub: Union[None, Subscriber] = None

        # Placeholders for ROS message data
        self.beacons: Union[None, np.ndarray] = None
        self.pose: Union[None, Tuple[float, ...]] = None

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
        self.odom_sub = ros.Subscriber(self.odom_sub_name, Odometry, callback=self.__odom_ros_sub,
                                       queue_size=self.QUEUE_SIZE)
        self.vel_pub = ros.Publisher(self.vel_pub_name, Twist, queue_size=self.QUEUE_SIZE)
        time.sleep(1)

    def run(self):
        # INITIAL GUESS
        particle_set = self.initialize_particleset()

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
        # Set placeholder variables mu_t and Sigma_t to our initial guess, and plot

        self.fig.plot_state_estimate(particle_set)

        for u_index in range(U.shape[0]):
            if ros.is_shutdown():
                break

            # Get the next input command
            u_t = U[u_index][:, None]

            # Send the input command to the robot
            self.__send_input_command(u_t=U[u_index], duration=dt)

            # Get the measurement
            y_t = np.array(list(self.beacons))[:, None]

            # Apply the Kalman filter
            particle_set = self.particle_filter(particle_set, u_t, y_t)

            self.fig.plot_state_estimate(particle_set)

        self.fig.stop()
        self.__send_input_command([0, 0])

    def initialize_particleset(self, N=1000):
        """

        :param N:
        :return:
        """
        particleset = {"mu_robot": np.zeros((N, 3, 1)),
                       "weights": np.full((N,), 1 / N),
                       "mu_beacons": np.zeros((N, 4, 2, 1)),
                       "Sigma_beacons": (np.identity(2)[None, None, ...] * 3.5 ** 2
                                         ).repeat(4, axis=1).repeat(N, axis=0)}

        return particleset

    def particle_filter(self, particle_set, u, y, siga=0.03, sigb=0.15):
        """
        Thus now a particle is composed of:
        > sampled robot position
        > for each beacon separately: the parameters mu and Sigma of its Gaussian estimate
          (2-vector, 2x2 matrix)
        > weight

        We suggest a dictionary where:
        > particle_set["mu_robot"][k]        contains sampled robot pose
        > particle_set["weights"][k]         contains the weight
        > particle_set["mu_beacons"][k, i]   contains beacon i mean position
        > particle_set["Sigma_beacons"][k,i] contains beacon i covariance matrix

        :param siga: standard deviation of motion noise
        :param sigb: standard devation of measurement noise
        """

        # *** YOUR CODE FOR EXERCISE 6B HERE ***
        nr_of_particles = particle_set['mu_robot'].shape[0]

        # We have some repeating terms in the motion model that are the same for every particle,
        # so we compute those here once, rather than on every iteration within the for-loop
        dx = (u[1] if u[0] == 0 else u[1] * sin(u[0]) / u[0]).squeeze()
        dy = 0 if u[0] == 0 else (u[1] * (1 - cos(u[0])) / u[0]).squeeze()

        for k in range(nr_of_particles):
            x_old = particle_set["mu_robot"][k]

            # Motion update: take its previous position, and update it according to one possibility
            # of motion; just fill in the equations of the motion model and add randomly
            # sampled noise where appropriate
            x_bar = np.array(
                [x_old[0] + cos(x_old[2]) * dx - sin(x_old[2]) * dy + np.random.normal(0, siga),
                 x_old[1] + sin(x_old[2]) * dx + cos(x_old[2]) * dy + np.random.normal(0, siga),
                 x_old[2] + u[0] + np.random.normal(0, siga ** 2)])

            # measurement update
            # sample random beacons
            beacon_II = list(range(4))
            random.shuffle(beacon_II)
            for y_ii, b_ii in enumerate(beacon_II):
                # Compute the weight
                mu_bar = particle_set['mu_beacons'][k][b_ii]
                y_bar = mu_bar - x_bar[:2]
                y_dev = y_bar - y[y_ii * 2:(y_ii * 2) + 2]
                Q = np.identity(2) * sigb ** 2
                LocSig = particle_set['Sigma_beacons'][k][b_ii] + Q
                # NB: here, the normalization of the Gaussian could be important (different one for
                #     each particle). So we have to do it.
                particle_set['weights'][k] *= \
                    np.exp(-0.5 * ((y_dev.T @ np.linalg.inv(LocSig) @ y_dev) / np.sqrt(
                        np.linalg.det(LocSig))).squeeze())

                # Update this beacon's Kalman filter
                # NB: here x is a constant, only beacon position is the variable!
                Sigma_bar = particle_set['Sigma_beacons'][k][b_ii]
                C = np.identity(2)
                D = -x_bar[:2]
                # The Kalman formulas
                K = Sigma_bar @ C.T @ np.linalg.inv(C @ Sigma_bar @ C.T + Q)
                mu = mu_bar + K @ (y[y_ii * 2:(y_ii * 2) + 2] - D - C @ mu_bar)
                Sigma = (np.identity(2) - K @ C) @ Sigma_bar

                # Nothing more to do: the result of these two steps is your new particle
                particle_set['mu_beacons'][k][b_ii] = mu
                particle_set['Sigma_beacons'][k][b_ii] = Sigma
            particle_set['mu_robot'][k] = x_bar

        # Normalize so all particles sum to one (now it actually matters for resampling)
        particle_set['weights'] /= np.sum(particle_set['weights'])
        # *** YOUR CODE END ***

        # *** YOUR CODE FOR EXERCISE 6C HERE ***
        # Generate new index list
        II = np.random.choice(np.arange(nr_of_particles), nr_of_particles,
                              p=particle_set['weights'])

        # generate new particle set
        particle_set["mu_robot"] = particle_set["mu_robot"][II]
        particle_set["weights"] = np.full_like(particle_set["weights"], 1 / nr_of_particles)
        particle_set["mu_beacons"] = particle_set["mu_beacons"][II]
        particle_set["Sigma_beacons"] = particle_set["Sigma_beacons"][II]
        # *** YOUR CODE END ***

        return particle_set

    def __send_input_command(self, u_t, duration=1.):
        for _ in range(1):
            msg = Twist()
            msg.angular.z = u_t[0]
            msg.linear.x = u_t[1]
            self.vel_pub.publish(msg)
            time.sleep(1.2)

    def __beacon_ros_sub(self, msg):
        self.beacons = msg.data

    def __odom_ros_sub(self, msg):
        quaternion = msg.pose.pose.orientation
        _, _, a = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, a)

    def __reset_simulation(self):
        """
        Reset simulation and wait a few seconds for Gazebo to catch up
        :return:
        """
        os.system("rostopic pub /reset std_msgs/Bool true -1")
        os.system("rosservice call /gazebo/reset_world")


class Figure:
    def __init__(self, title="State Estimate"):
        self.title = title

        self.X = np.stack(np.meshgrid(np.linspace(-1.25, 1.25, 125),
                                      np.linspace(-1.75, 1.75, 175)), axis=-1)
        self.theta = np.linspace(-np.pi, np.pi, 360)[:, None]

        self.__pipe_in, self.__pipe_ou = multiprocessing.Pipe()
        self.process: Union[None, multiprocessing.Process] = None

    def start(self):
        self.process = multiprocessing.Process(target=self.__run)
        self.process.start()
        atexit.register(cv2.destroyAllWindows)
        return self

    def stop(self):
        self.__pipe_in.send(None)
        self.process.join()

    def plot_state_estimate(self, particle_set):
        best_particle_index = np.argmax(particle_set["weights"])
        self.__pipe_in.send((particle_set["mu_robot"][best_particle_index],
                             particle_set["mu_beacons"][best_particle_index],
                             particle_set["Sigma_beacons"][best_particle_index]))

    def __run(self):
        # Create figure
        fig = plt.figure(figsize=(4.20, 8.00))
        ax0: Axes = plt.axes([0.1, 0.48, 0.8, 0.42])
        ax1: Axes = plt.axes([0.1, 0.1, 0.8, 0.28], polar=True)

        while True:
            msg = self.__pipe_ou.recv()
            if msg is None:
                self.__display_opencv_image(fig, duration=300)
                break
            mu_robot, mu_beacons, Sigma_beacons = msg

            ax0.clear(), ax1.clear()
            ax0.scatter(mu_robot[0], mu_robot[1], c=BLUE)
            ax0.set_xlim(-1.25, 1.25)
            ax0.set_ylim(-1.75, 1.75)
            ax0.set_aspect('equal')

            for ii in range(mu_beacons.shape[0]):
                self.__uncertainty_ellipse(ax0, mu_beacons[ii], Sigma_beacons[ii])

            ax1.plot([0, mu_robot[2]], [0, 1], c=BLUE, linewidth=5)

            retval = self.__display_opencv_image(fig)
            if not retval:
                break

        cv2.destroyAllWindows()

    def __uncertainty_ellipse(self, ax, mu, Sigma):
        ax.autoscale(False)
        w, v = np.linalg.eigh(Sigma)
        el = Ellipse(xy=mu, width=3 * np.sqrt(w[0]), height=3 * np.sqrt(w[1]),
                     angle=np.arctan2(v[1, 0], v[0, 0]), color=GREEN, linewidth=2, fill=False)
        ax.add_patch(el)

    def __display_opencv_image(self, figure, duration=0.1):
        figure.canvas.draw()
        img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.title, img)

        t0 = time.time()
        cv2.waitKey(50)
        while not ros.is_shutdown():
            try:
                cv2.getWindowProperty(self.title, 0)
                cv2.waitKey(50)
                if time.time() - t0 > duration:
                    return True
            except cv2.error:
                return False
        return True


if __name__ == '__main__':
    f = SLAMParticleFilter()
    f.start_ros()
    f.run()
