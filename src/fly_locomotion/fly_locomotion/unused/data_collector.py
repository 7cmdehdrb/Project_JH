# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np
from loguru import logger


class DataCollectNode(Node):
    def __init__(self):
        super().__init__("data_collect_node")

        self.__skeleton_sub = self.create_subscription(
            PoseArray,
            "/l_hand_skeleton_pose",
            self.__skeleton_callback,
            qos_profile=qos_profile_system_default,
        )

        logger.add(
            "/home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/dataskeleton_data2.txt",
            format="{message}",
        )

        self.__cls = "else"

    def __skeleton_callback(self, msg: PoseArray):
        def stringfy(msg: PoseArray) -> str:
            s = ""
            for pose in msg.poses:
                pose: Pose
                s += f"{pose.position.x} {pose.position.y} {pose.position.z} {pose.orientation.x} {pose.orientation.y} {pose.orientation.z} {pose.orientation.w} "
            return s.strip()

        txt_data = self.__cls + " " + stringfy(msg)
        logger.info(txt_data)


def main():
    rclpy.init(args=None)

    node = DataCollectNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
