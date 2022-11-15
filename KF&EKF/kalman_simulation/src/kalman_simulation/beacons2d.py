#!/usr/bin/env python

import random
import time
from typing import Union

import numpy as np

import rospy as ros
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetWorldProperties, GetModelState
from rospy import Publisher, Subscriber
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


class Beacons2D:
    """
    This class represents a simplified mannequin detection node that uses gazebo simulation data to
    publish mannequin locations with an accuracy and publish rate that would not be feasible on the real
    robot. This class ignores several important features of reality; for example, mannequins need not
    be visible to the robots sensors for their locations to be published, and orientation of the turtlebot
    is ignored.
    """
    NOISE_STDDEV = 0.25
    PUBLISH_RATE: int = 10
    QUEUE_SIZE: int = 2

    def __init__(self, name: str = "beacons2d"):
        self.node_name: str = name

        # Declare ROS subscriber names
        self.gazebo_sub_name: str = "/gazebo/model_states"
        self.gazebo_sub: Union[None, Subscriber] = None

        # Declare ROS publisher names
        self.mannequin_pub_name: str = "/mannequins"
        self.mannequin_pub: Union[None, Publisher] = None

        # Placeholder for gazebo model state
        self.model_states = {}

    def start_ros(self) -> None:
        """
        Initialize ros and set all Subscribers and Publishers
        :return:
        """
        ros.init_node(self.node_name, log_level=ros.INFO)

        self.gazebo_sub = ros.Subscriber(self.gazebo_sub_name, ModelStates, callback=self.__gazebo_ros_sub,
                                         queue_size=self.QUEUE_SIZE)
        self.mannequin_pub = ros.Publisher(self.mannequin_pub_name, numpy_msg(Floats), queue_size=self.QUEUE_SIZE)

    def run(self):
        """
        The main loop repeatedly calculates the relative euclidean position of mannequins to the turtlebot
        and publishes these in an array with fixed order:
        [mannequin_NW_x, mannequin_NW_y, mannequin_NE_x, ..., mannequin_SW_x, mannequin_SW_y]
        :return:
        """
        while True:
            time.sleep(1 / self.PUBLISH_RATE)
            try:
                # Get the absolute location of the turtlebot
                turtle_x = self.model_states['turtlebot3_omni']['pose'].position.x
                turtle_y = self.model_states['turtlebot3_omni']['pose'].position.y

                # Get the aboslute locations of the mannequins
                mannequins = []
                for mannequin in ('mannequin_NW', 'mannequin_NE', 'mannequin_SE', 'mannequin_SW'):
                    # Determine relative position
                    mannequins.append(self.model_states[mannequin]['pose'].position.x - turtle_x
                                   + random.gauss(0, self.NOISE_STDDEV))
                    mannequins.append(self.model_states[mannequin]['pose'].position.y - turtle_y
                                   + random.gauss(0, self.NOISE_STDDEV))

                # Publish mannequin locations to /mannequins
                mannequins = np.array(mannequins, dtype=np.float32)
                self.mannequin_pub.publish(mannequins)

            except (KeyError, TypeError):
                # Some data is not available, try again next tick
                pass

    def __gazebo_ros_sub(self, msg):
        """
        Callback function when a new gazebo model state comes in
        :param msg:
        :return:
        """
        for key in self.model_states:
            if key not in msg.name: # if state of a model is unavailable, set it to None
                self.model_states[key] = None

        for ii, model in enumerate(msg.name): # store models location
            self.model_states[model] = {'pose': msg.pose[ii], 'twist': msg.twist[ii]}


def main():
    publisher = Beacons2D()
    publisher.start_ros()
    publisher.run()

if __name__ == '__main__':
    main()
