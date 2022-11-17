import atexit
import multiprocessing
import os
import time
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import sin, cos

import rospy as ros
from geometry_msgs.msg import Twist

from rospy import Subscriber
from rospy_tutorials.msg import Floats
from tf.transformations import euler_from_quaternion

BLUE = "#1E64C8"
GREEN = "#71A860"


class ParticleFilter:
    """
    This class implements SLAM with a particle filter. Requirement for this is that the robot never
    confuses two beacons
    """
    QUEUE_SIZE = 2

    def __init__(self, name: str = "slam_particle_filter"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.beacon_sub_name: str = "/mannequins"
        self.beacon_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        # We use a hacky model of the robot, so commands can go straight to /cmd_vel (inspect the
        # launch files to see what is going on)
        self.vel_pub_name: str = "/cmd_vel"
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
                      [trn, 0.1], [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1],
                      [trn, 0.1], [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1], [0., 0.2],
                      [trn, 0.1], [trn, 0.1], [trn, 0.2], [trn, 0.2], [trn, 0.1],
                      [-trn, 0.1], [-trn, 0.1], [-trn, 0.2], [-trn, 0.2], [-trn, 0.1], [0., 0.2],
                      [-trn, 0.1], [-trn, 0.1], [-trn, 0.2], [-trn, 0.2], [-trn, 0.1],
                      [-trn, 0.1], [-trn, 0.1], [-trn, 0.2], [-trn, 0.2], [-trn, 0.1], [0., 0.2],
                      [-trn, 0.1], [-trn, 0.1], [-trn, 0.2], [-trn, 0.2], [-trn, 0.1],
                      [0., 0.0], [0., 0.0], [0., 0.0], [0., 0.0], [0., 0.0]])
        dt = 1  # seconds

        # CHANGE THESE PARAMETERS TO USE A DIFFERENT NUMBER OF BEACONS AND PARTICLES
        number_of_beacons = 2  # in {1, 2, 3, 4} # change this value to work with fewer beacons
        number_of_particles = 3000  # more particles give better estimates but slower performance

        # INITIAL GUESS
        particle_set = self.__initialize_particleset(number_of_beacons, number_of_particles)
        # plot initial guess
        self.fig.plot_state_estimate(particle_set)

        for u_index in range(U.shape[0]):
            if ros.is_shutdown():
                break

            # Get the next input command
            u_t = U[u_index][:, None]

            # Send the input command to the robot
            self.__send_input_command(u_t=U[u_index], duration=dt)

            # Get the measurement
            y_t = np.array(list(self.beacons))[:number_of_beacons * 2, None]

            # Apply the Particle filter
            particle_set = self.particle_filter(particle_set, u_t, y_t, number_of_beacons)

            self.fig.plot_state_estimate(particle_set)

        self.fig.stop()
        self.__send_input_command([0, 0])

    def particle_filter(self, particle_set, u_t, y_t, number_of_beacons, siga=0.01, sigb=0.15):
        """
        Performs a single update of all particles in the particle set.

        :param particle_set: contains your set of N particles. The particle set has been
                             implemented as a dictionary: a hash table that you query with strings
                             (usually).
               > particle_set["particles"] contains robot pose and beacon positions, mirroring the
                 manner in which we have defined our state in the Kalman filter before: this is a
                 (number_of_particles)x(3+2*number_of_beacons) matrix
               > particle_set["weights"] containts weights (vector of length number_of_particles)
        :param u_t: the input command at time t (vector of length 2)
        :param y_t: the measurement data at time t (vector of length 2*number_of_beacons)
        :param number_of_beacons: how many beacons you want to use
        :param siga: standard deviation motion noise
        :param sigb: standard deviation sensor noise
        :return:
        """

        # How many beacons you try to use and map. In principle this is 4, but start out with just
        # one and then see how adding more particles affects performance.
        Nbb = number_of_beacons

        # HERE THE SOLUTION
        N = particle_set["particles"].shape[0]
        for k in range(N):
            # Get the next particle
            x_old = particle_set["particles"][k]

            # 1. Motion update step: defining xbar_k
            # take the particle's old position, and update it according to one possibility of
            # motion.
            # tip 1: beacon positions don't move
            # tip 2: the implementation of this step is almost identical to how we computed mu_bar
            #        in exercise 2. The difference is that we no longer ignore noise; instead, add
            #        a random noise sample with np.random.normal(0, siga ** 2) at the appropriate
            #        positions in the motion equations.

            

            # *** YOUR CODE START ***
            x_bar = x_old 

            dx = (u_t[1] if u_t[0] == 0 else u_t[1] * sin(u_t[0]) / u_t[0]).squeeze()           #Squeeze is used for dim. reduction (ex. 1,3,3 -> 3,3)
            dy = 0 if u_t[0] == 0 else (u_t[1] * (1 - cos(u_t[0])) / u_t[0]).squeeze()
            x_bar[:3] = np.array(
                [x_old[0] + cos(x_old[2]) * dx - sin(x_old[2]) * dy + np.random.normal(0, siga),
                 x_old[1] + sin(x_old[2]) * dx + cos(x_old[2]) * dy + np.random.normal(0, siga),
                 x_old[2] + u_t[0] + np.random.normal(0, siga ** 2)])

            # *** YOUR CODE END ***

            particle_set["particles"][k] = x_bar

            # 2. Sensor update step
            # take the weight of the particle, and the measurement y_t. Multiply the weight by
            # p(y | xbar_k) -- this is the Gaussian function exp(-0.5 * (y_bar - y)**2 / sigb**2),
            # where y_bar is the expected measurement given xbar_k
            # tip 1: the Gaussian for 8 independent measurements is just the product of the
            #        standard 1-D Gaussians.
            # tip 2: Because we sample a fixed trajectory for each particle, the beacon positions
            #        (and corresponding measurements) are independent. So instead of computing
            #        y_bar all at once - like we do with the Kalman filter - you could add another
            #        for-loop in here and process the measurements one by one.
            # tip 3: normalization (i.e. making sure that the gaussian sums to 1) is totally
            #        unimportant in this case, as it is the same value for every measurement, and
            #        we only want to *compare* the options. You can drop it.

            

            # *** YOUR CODE START ***
            #y_bar = x_bar[3:] - np.repeat(x_bar[:2], Nbb, axis=0)
            
            for n in range(Nbb):
                y_bar = x_bar[n+3] - x_bar[n]

            P = np.exp(-0.5 * (y_bar - y_t)**2 / sigb**2)  # P(y | xbar_k)
            P = np.prod(P)

            # *** YOUR CODE END ***

            particle_set["weights"][k] *= P  # multiply the weight by P(y | xbar_k)

        # 3. Nothing more to do: the result of these two steps is your new particle set
        # A more efficient PF would be to do resampling now, but we skip it for the moment

        # Normalize weights. Mathematically, there is no need for this, but weights can become
        # very small or very large and Python might round them to 0 or infinity if we do not do
        # this step.
        particle_set["weights"] = particle_set["weights"] / np.sum(particle_set["weights"])

        return particle_set

    def __initialize_particleset(self, n_beacons, n_particles):
        particleset = {"particles": np.zeros((n_particles, 3 + 2 * n_beacons, 1)),
                       "weights": np.full((n_particles,), 1 / n_particles)}

        # X1-coordinates of beacons between -1.25 and 1.25
        particleset["particles"][:, 3::2] = \
            np.random.uniform(-1.25, 1.25, (n_particles, n_beacons, 1))
        # x2-coordinates of beacons between -1.75 and 1.75
        particleset["particles"][:, 4::2] = \
            np.random.uniform(-1.75, 1.75, (n_particles, n_beacons, 1))

        return particleset

    def __send_input_command(self, u_t, duration=1.):
        msg = Twist()
        msg.angular.z = u_t[0]
        msg.linear.x = u_t[1]
        self.vel_pub.publish(msg)
        time.sleep(1)

    def __beacon_ros_sub(self, msg):
        self.beacons = msg.data

    def __odom_ros_sub(self, msg):
        quaternion = msg.pose.pose.orientation
        _, _, a = euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
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
        self.__pipe_in.send(particle_set["particles"][best_particle_index])

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
            mu = msg

            ax0.clear(), ax1.clear()
            ax0.scatter(mu[0], mu[1], c=BLUE)
            ax0.scatter(mu[3::2], mu[4::2], c=GREEN, marker='x')
            ax0.set_xlim(-1.25, 1.25)
            ax0.set_ylim(-1.75, 1.75)
            ax0.set_aspect('equal')

            ax1.plot([0, mu[2]], [0, 1], c=BLUE, linewidth=5)

            if not self.__display_opencv_image(fig):
                break

        cv2.destroyAllWindows()

    def __display_opencv_image(self, figure, duration=0.1):
        figure.canvas.draw()
        img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.title, img)

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
    pf = ParticleFilter()
    pf.start_ros()
    pf.run()
