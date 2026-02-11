# ROS2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# Message
import rclpy.time
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *

import time

class Gesture(Node):
    def __init__(self):
        super().__init__("test")
        current_time = time.strftime("%Y%m%d_%H%M%S")
        self.file = open(f"/home/hoon/hand_skeleton/fly_locomotion/data_save/test_{current_time}.txt", "w")

        self.create_subscription(
            msg_type=PoseStamped,
            topic='player_axis_data',
            callback=self.player_axis_data_callback,
            qos_profile=QoSProfile(depth=10)
        )

        self.player_axis_position = Point()
        self.player_axis_orientation = Quaternion()

        self.create_subscription(
            msg_type=PoseStamped,
            topic='target_axis_data',
            callback=self.target_axis_data_callback,
            qos_profile=QoSProfile(depth=10)
        )

        self.target_axis_position = Point()
        self.target_axis_orientation = Quaternion()

    def player_axis_data_callback(self, msg):
        self.player_axis_position = msg.pose.position
        self.player_axis_orientation = msg.pose.orientation

        self.file.write(f"{self.player_axis_position.x}, {self.player_axis_position.y}, {self.player_axis_position.z}, "
                        f"{self.player_axis_orientation.x}, {self.player_axis_orientation.y}, "
                        f"{self.player_axis_orientation.z}, {self.player_axis_orientation.w}, "
                        f"{self.target_axis_position.x}, {self.target_axis_position.y}, {self.target_axis_position.z}, "
                        f"{self.target_axis_orientation.x}, {self.target_axis_orientation.y}, "
                        f"{self.target_axis_orientation.z}, {self.target_axis_orientation.w}\n")

    def target_axis_data_callback(self, msg):
        self.target_axis_position = msg.pose.position
        self.target_axis_orientation = msg.pose.orientation
def main():
    rclpy.init()
    node = Gesture()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
