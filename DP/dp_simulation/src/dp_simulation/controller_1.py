import random
import time
from typing import Union

import numpy as np

import rospy as ros
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Quaternion
from rospy import Subscriber, Publisher, ServiceProxy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class Controller:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 100

    def __init__(self, name: str = "dp_controller"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.gzb_sub_name: str = "/gazebo/model_states"
        self.gzb_sub: Union[None, Subscriber] = None
        self.sys_sub_name: str = "/syscommand"
        self.sys_sub: Union[None, Subscriber] = None
        self.nav_sub_name: str = "/nav_planner"
        self.nav_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.vel_pub_name: str = "/cmd_vel"
        self.vel_pub: Union[None, Publisher] = None

        # Declare ROS service
        ros.wait_for_service("/gazebo/set_model_state")
        self.set_state: ServiceProxy = ServiceProxy("/gazebo/set_model_state", SetModelState)

        # Placeholders
        self.U, self.model_states = None, {}
        self.condition = 0  #

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.gzb_sub = ros.Subscriber(self.gzb_sub_name, ModelStates, callback=self.__gzb_ros_sub,
                                      queue_size=self.QUEUE_SIZE)
        self.sys_sub = ros.Subscriber(self.sys_sub_name, String, callback=self.__sys_ros_sub,
                                      queue_size=self.QUEUE_SIZE)
        self.nav_sub = ros.Subscriber(self.nav_sub_name, numpy_msg(Floats),
                                      callback=self.__nav_ros_sub, queue_size=self.QUEUE_SIZE)

        self.vel_pub = ros.Publisher(self.vel_pub_name, Twist, queue_size=self.QUEUE_SIZE)
        time.sleep(1)

    def run(self):
        self.__init_or_reset_world()

        t_moved = time.time()

        while not ros.is_shutdown():
            try:
                xreal, yreal, treal = self.get_robot_pose()
                xgrid, ygrid, tgrid = self.real2grid(xreal, yreal, treal)
                x_tgt, y_tgt = self.get_target_square(xgrid, ygrid, tgrid)

                th_cmd = self.compute_angular_cmd(xreal, yreal, treal,
                                                  *self.grid2real(x_tgt, y_tgt))
                v_cmd = self.compute_linear_cmd(xreal, yreal, treal, *self.grid2real(x_tgt, y_tgt))

                self.publish_vel_cmd(v_cmd, th_cmd)

                if th_cmd > 0.01 or v_cmd > 0.01:
                    t_moved = time.time()
                elif time.time() - t_moved > 5:
                    self.__init_or_reset_world()
                    t_moved = time.time()
            except Exception:
                time.sleep(1 / self.PUBLISH_RATE)

    def __init_or_reset_world(self):
        while not any("turtlebot3" in model for model in self.model_states.keys()):
            time.sleep(0.1)

        xgrid, ygrid = random.randint(0, 6), random.randint(0, 9)
        while (xgrid, ygrid) in [(0, 1), (2, 1), (6, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8)]:
            xgrid, ygrid = random.randint(0, 6), random.randint(0, 9)

        xreal, yreal = self.grid2real(xgrid, ygrid)
        theta = random.choice([-0.5 * np.pi, 0, 0.5 * np.pi, np.pi])
        self.teleport(xreal, yreal, theta)

    def __gzb_ros_sub(self, msg):
        for key in self.model_states:
            if key not in msg.name:
                self.model_states[key] = None

        for ii, model in enumerate(msg.name):
            self.model_states[model] = {'pose': msg.pose[ii], 'twist': msg.twist[ii]}

    def __sys_ros_sub(self, msg):
        if msg.data == "reset":
            self.__init_or_reset_world()

    def __nav_ros_sub(self, msg):
        try:
            self.U = msg.data.reshape((10, 7, 2))
        except ValueError:
            ros.logerr(f"Size of message ({msg.data.shape[0]}) does not fit expected shape "
                       f"(10, 7, 2)")
            ros.logerr(f"Policy not updated; check that U is correct shape.")

    def publish_vel_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.vel_pub.publish(msg)
        time.sleep(1 / self.PUBLISH_RATE)

    def grid2real(self, x, y):
        W, H = 2.5, 3.5
        w, h = W / 7, H / 10

        xreal = (-3 + x) * w
        yreal = (4.5 - y) * h

        return xreal, yreal

    def real2grid(self, x, y, theta):
        tgrid = round((theta % (2 * np.pi)) / (0.5 * np.pi))

        W, H = 2.5, 3.5
        w, h = W / 7, H / 10

        xgrid = round(x / w + 3)
        ygrid = round(4.5 - y / h)

        return xgrid, ygrid, tgrid

    def get_robot_pose(self):
        xreal = self.model_states['turtlebot3_christmas']["pose"].position.x
        yreal = self.model_states['turtlebot3_christmas']["pose"].position.y
        qreal = self.model_states['turtlebot3_christmas']["pose"].orientation
        _, _, treal = euler_from_quaternion([qreal.x, qreal.y, qreal.z, qreal.w])
        return xreal, yreal, treal

    def get_target_square(self, x, y, t):
        if self.U is None:
            return x, y

        return self.U[y, x, 1], self.U[y, x, 0]

    def compute_angular_cmd(self, x, y, angle, x_tgt, y_tgt):
        if np.isclose(x, x_tgt, rtol=0.01, atol=0.02) \
                and np.isclose(y, y_tgt, rtol=0.01, atol=0.02):
            return 0

        angle_tgt = np.arctan2(y_tgt - y, x_tgt - x)
        error = (angle_tgt - angle + np.pi) % (2 * np.pi) - np.pi
        cmd = np.clip(error * 2, -2.84, 2.84)
        return cmd

    def compute_linear_cmd(self, x, y, angle, x_tgt, y_tgt):
        angle_tgt = np.arctan2(y_tgt - y, x_tgt - x)
        error = (angle_tgt - angle + np.pi) % (2 * np.pi) - np.pi
        rel_err = abs(error / (0.5 * np.pi))
        cmd = 0.22 * np.clip(1 - rel_err, 0, 1)
        return cmd

    def teleport(self, x, y, theta=0.):
        msg = ModelState()

        msg.model_name = "turtlebot3_christmas"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.02

        msg.pose.orientation = Quaternion(*quaternion_from_euler(0., 0., theta))
        self.set_state(msg)
        time.sleep(0.1)


def main():
    ros.loginfo("HELLO")
    try:
        c = Controller()
    except:
        ros.loginfo("init")
    try:
        c.start_ros()
    except:
        ros.loginfo("start")
    try:
        c.run()
    except:
        ros.loginfo("run")


if __name__ == '__main__':
    main()
