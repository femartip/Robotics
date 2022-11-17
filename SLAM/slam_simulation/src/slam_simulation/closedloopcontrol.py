import time
from collections import deque
from typing import Union

import numpy as np

import rospy as ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from rospy import Subscriber, Publisher
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion


class ClosedLoopControl:
    """
    This class implements closed loop control for the turtlebot3. You can send Twist() velocity
    commands to the /closedloop_cmd topic, similar to how you would to the /cmd_vel topic.
    This class will send the appropriate command to /cmd_vel, correcting for noise on the angular
    velocity using the data in the /odom topic.
    """
    QUEUE_SIZE = 2
    PUBLISH_RATE = 100

    def __init__(self, name: str = "closed_loop_controller"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.odom_sub_name: str = "/odom"
        self.odom_sub: Union[None, Subscriber] = None
        self.cmd_sub_name: str = "/closedloop_cmd"
        self.cmd_sub: Union[None, Subscriber] = None
        self.reset_sub_name: str = "/reset"
        self.reset_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.vel_pub_name: str = "/cmd_vel"
        self.vel_pub: Union[None, Publisher]

        # Initial angular and linear speed
        self.a0, self.a, self.da, self.dx, self.t0 = 0, 0, 0, 0, time.time()
        self.__init_or_reset_world()

    def start_ros(self):
        """
        Initialise node, subscribers, and publishers.
        :return:
        """
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.odom_sub = ros.Subscriber(self.odom_sub_name, Odometry, callback=self.__odom_ros_sub,
                                       queue_size=self.QUEUE_SIZE)
        self.cmd_sub = ros.Subscriber(self.cmd_sub_name, Twist, callback=self.__cmd_ros_sub,
                                      queue_size=self.QUEUE_SIZE)
        self.reset_sub = ros.Subscriber(self.reset_sub_name, Bool, callback=self.__reset_ros_sub,
                                        queue_size=self.QUEUE_SIZE)

        self.vel_pub = ros.Publisher(self.vel_pub_name, Twist, queue_size=self.QUEUE_SIZE)

    def run(self):
        P = 5 # We use a P controller, which is sufficient
        while not ros.is_shutdown():
            # compute the orientation the turtlebot3 should be facing
            dt = time.time() - self.t0
            a_tgt = self.a0 + dt * self.da

            # compute error with actual heading
            a_err = (a_tgt - self.a + np.pi) % (2 * np.pi) - np.pi

            # send cmd_vel correcting for angular deviation
            msg = Twist()
            msg.angular.z = np.clip(self.da + P * a_err, -2.84, 2.84)
            msg.linear.x = np.clip(self.dx, 0., 0.22)
            self.vel_pub.publish(msg)

            time.sleep(1 / self.PUBLISH_RATE)

    def __init_or_reset_world(self):
        self.a0, self.a, self.da, self.dx, self.t0 = 0, 0, 0, 0, time.time()
        self.cmd_buffer = deque([])

    def __odom_ros_sub(self, msg):
        """
        Callback function for odometry data
        :return:
        """
        quaternion = msg.pose.pose.orientation
        _, _, self.a = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])

    def __cmd_ros_sub(self, msg: Twist):
        """
        Callback function for
        :param msg:
        :return:
        """
        self.a0 = self.a
        self.da = msg.angular.z
        self.dx = msg.linear.x
        self.t0 = time.time()

    def __reset_ros_sub(self, msg):
        """
        Callback function for simulation reset message
        :param msg:
        :return:
        """
        self.__init_or_reset_world()


def main():
    control = ClosedLoopControl()
    control.start_ros()
    control.run()


if __name__ == '__main__':
    main()
