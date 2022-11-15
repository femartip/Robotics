import random
import time
from typing import Union

import numpy as np

import rospy as ros
from gazebo_msgs.msg import ModelStates
from rospy import Subscriber, Publisher
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


class Beacons1D:
    QUEUE_SIZE = 2
    PUBLISH_RATE = 10
    NOISE_STDDEV = 0.2

    def __init__(self, name: str = "beacons1d"):
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
        while not ros.is_shutdown():
            time.sleep(1 / self.PUBLISH_RATE)
            try:
                # Get the absolute location of the turtlebot
                turtle_x = self.model_states['turtlebot3_omni']['pose'].position.x

                # Get the aboslute locations of the mannequins
                mannequins = []
                for mannequin in ('lonely_person',):
                    # Determine relative position
                    mannequins.append(self.model_states[mannequin]['pose'].position.x - turtle_x
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
    publisher = Beacons1D()
    publisher.start_ros()
    publisher.run()

if __name__ == '__main__':
    main()
