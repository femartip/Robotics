import multiprocessing
import os
import math as m
import time
from collections import deque
from typing import Union

import numpy as np
from numpy import pi, cos, sin

import rospy as ros
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Vector3, Twist
from rospy import Subscriber, ServiceProxy, Duration
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion


class OmniWheelController:
    NOISE_STDDEV = 0.03
    ORIENTATION = 0. * pi  # rad

    def __init__(self, x0=0., y0=0., ang0=0., name: str = "omniwheel_controller"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.gazebo_sub_name: str = "/gazebo/model_states"
        self.gazebo_sub: Union[None, Subscriber] = None
        self.cmd_sub_name: str = "/cmd_vel"
        self.cmd_sub: Union[None, Subscriber] = None
        self.reset_sub_name: str = "/reset"
        self.reset_sub: Union[None, Subscriber] = None

        # Declare ROS service
        self.apply_wrench: ServiceProxy = ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        self.pub_rate: int = 120
        self.queue_size: int = 2

        self.model_states = {}

        # Initial position and speed
        self.x0, self.y0, self.ang0 = x0, y0, ang0
        self.dx, self.da = 0, 0
        self.__init_or_reset_world()

    def __init_or_reset_world(self):
        self.x_prev, self.y_prev, self.ang_prev, self.t_prev = self.x0, self.y0, self.ang0, time.time()
        self.x_tgt, self.y_tgt, self.ang_tgt = self.x0, self.y0, self.ang0
        self.x, self.y, self.ang = self.x0, self.y0, self.ang0
        self.dx, self.da = 0, 0
        self.cmd_buffer = deque([])

    def start_ros(self):
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.gazebo_sub = ros.Subscriber(self.gazebo_sub_name, ModelStates, callback=self.__gazebo_ros_sub,
                                         queue_size=self.queue_size)
        self.cmd_sub = ros.Subscriber(self.cmd_sub_name, Twist, callback=self.__cmd_ros_sub,
                                      queue_size=self.queue_size)
        self.reset_sub = ros.Subscriber(self.reset_sub_name, Bool, callback=self.__reset_ros_sub,
                                        queue_size=self.queue_size)

    def update_target_pos(self):
        t = time.time()
        if t < self.t_prev + 1:
            dt = t - self.t_prev

            if self.da == 0:
                self.x = self.x_prev + cos(self.ang_prev) * self.dx * dt
                self.y = self.y_prev + sin(self.ang_prev) * self.dx * dt
            else:
                self.x = self.x_prev + cos(self.ang_prev) * self.dx * sin(self.da * dt) / self.da \
                         - sin(self.ang_prev) * self.dx * (1 - cos(self.da * dt)) / self.da
                self.y = self.y_prev + sin(self.ang_prev) * self.dx * sin(self.da * dt) / self.da \
                         + cos(self.ang_prev) * self.dx * (1 - cos(self.da * dt)) / self.da

            da = (self.ang_tgt - self.ang_prev + pi) % (2 * pi) - pi

            self.ang = self.ang_prev + dt * da
        else:
            self.x_prev, self.y_prev, self.ang_prev = self.x_tgt, self.y_tgt, self.ang_tgt
            self.x, self.y, self.ang = self.x_tgt, self.y_tgt, self.ang_tgt
            if len(self.cmd_buffer) > 0:
                dx, da = self.cmd_buffer[0]
                self.cmd_buffer.popleft()
                self.dx, self.da = dx, da

                if da == 0:
                    self.x_tgt = self.x_prev + cos(self.ang_prev) * dx
                    self.y_tgt = self.y_prev + sin(self.ang_prev) * dx
                else:
                    self.x_tgt = self.x_prev + cos(self.ang_prev) * dx * sin(da) / da - sin(self.ang_prev) * dx * (1 - cos(da)) / da
                    self.y_tgt = self.y_prev + sin(self.ang_prev) * dx * sin(da) / da + cos(self.ang_prev) * dx * (1 - cos(da)) / da




                self.ang_tgt = (self.ang_prev + da) % (2 * pi)
                self.t_prev = time.time()

    def run(self):
        while not ros.is_shutdown():
            try:
                quaternion = self.model_states['turtlebot3_omni']['pose'].orientation
                _, _, self.ang_prev = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
                self.x_prev = self.model_states['turtlebot3_omni']['pose'].position.x
                self.y_prev = self.model_states['turtlebot3_omni']['pose'].position.y
                self.t_prev = time.time()
                break
            except (ros.service.ServiceException, KeyError):
                pass

        buffer_yaw = np.zeros((2,), dtype=np.float32)
        P_yaw, D_yaw = .1, 10.

        buffer_x = np.zeros((2,), dtype=np.float32)
        P_x, D_x = 10., 2000.

        buffer_y = np.zeros((2,), dtype=np.float32)
        P_y, D_y = 10., 2000.

        t0 = time.time()
        while not ros.is_shutdown():
            try:
                self.update_target_pos()

                # YAW
                quaternion = self.model_states['turtlebot3_omni']['pose'].orientation
                _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
                yaw_error = self.ang - yaw
                yaw_error = (yaw_error + pi) % (2 * pi) - pi

                buffer_yaw = np.roll(buffer_yaw, shift=1)
                buffer_yaw[0] = yaw_error
                F_yaw = P_yaw * buffer_yaw[0] + D_yaw * (buffer_yaw[0] - buffer_yaw[1])

                # POSITION
                x_err = self.x - self.model_states['turtlebot3_omni']['pose'].position.x
                buffer_x = np.roll(buffer_x, shift=1)
                buffer_x[0] = x_err

                y_err = self.y - self.model_states['turtlebot3_omni']['pose'].position.y
                buffer_y = np.roll(buffer_y, shift=1)
                buffer_y[0] = y_err

                x_real = self.model_states['turtlebot3_omni']['pose'].position.x
                y_real = self.model_states['turtlebot3_omni']['pose'].position.y

                F_x = P_x * buffer_x[0] \
                      + D_x * (buffer_x[0] - buffer_x[1])

                F_y = P_y * buffer_y[0] \
                      + D_y * (buffer_y[0] - buffer_y[1])
                # print(f"({x_real:.3f}, {F_x:.3f}, {x_err:.3f})")

                wrench = Wrench(force=Vector3(F_x, F_y, 0.),
                                torque=Vector3(0., 0., F_yaw))

                if time.time() > t0 + 0.1:
                    self.apply_wrench(body_name="turtlebot3_omni::base_footprint",
                                      duration=Duration(nsecs=int(1 / self.pub_rate * 1e9)),
                                      wrench=wrench)
                time.sleep(1 / self.pub_rate)

            except KeyError as ke:
                pass
            except ros.service.ServiceException:
                pass

    def __gazebo_ros_sub(self, msg):
        for key in self.model_states:
            if key not in msg.name:
                self.model_states[key] = None

        for ii, model in enumerate(msg.name):
            self.model_states[model] = {'pose': msg.pose[ii], 'twist': msg.twist[ii]}

    def __cmd_ros_sub(self, msg):
        self.cmd_buffer.append((msg.linear.x, msg.angular.z))

    def __reset_ros_sub(self, msg):
        """

        :param msg:
        :return:
        """
        self.__init_or_reset_world()


def main(x0=0, y0=0, ang0=0):
    c = OmniWheelController(x0, y0, ang0)
    c.start_ros()
    c.run()


if __name__ == '__main__':
    main()
