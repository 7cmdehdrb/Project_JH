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

# TF
from tf2_ros import *

import os
import time
from loguru import logger


class ParameterNode(Node):
    def __init__(self):
        super().__init__("parameter_node")

        self.passed_waypoint = 0
        self.linear_velocity_gain = 0.0
        self.angular_velocity_gain = 0.0

        self.passed_waypoint_sub = self.create_subscription(
            Int32,
            "passed_waypoint_count",
            self.passed_waypoint_callback,
            qos_profile_system_default,
        )

        self.lin_vel_gain_sub = self.create_subscription(
            Float32,
            "linear_velocity_gain",
            self.lin_vel_gain_callback,
            qos_profile_system_default,
        )

        self.ang_vel_gain_sub = self.create_subscription(
            Float32,
            "angular_velocity_gain",
            self.ang_vel_gain_callback,
            qos_profile_system_default,
        )

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        file_name = time.strftime("%Y-%m-%d-%H-%M-%S_parameter") + ".csv"
        logger.add(  # TODO: Change Sub-Folder name for each users
            os.path.join(ROOT_DIR, "parameter_test/CHG" "", file_name),
            format="{message}",
        )
        logger.info("linear_velocity_gain,angular_velocity_gain,passed_waypoint,time")

    def passed_waypoint_callback(self, msg):
        self.passed_waypoint = msg.data

    def lin_vel_gain_callback(self, msg):
        self.linear_velocity_gain = msg.data

    def ang_vel_gain_callback(self, msg):
        self.angular_velocity_gain = msg.data


def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()
    import threading

    th = threading.Thread(target=rclpy.spin, args=(parameter_node,), daemon=True)
    th.start()
    start_time = time.time()

    try:
        r = parameter_node.create_rate(100.0)
        while rclpy.ok():
            r.sleep()
    except KeyboardInterrupt:
        end_time = time.time()
        logger.info(
            f"{parameter_node.linear_velocity_gain},{parameter_node.angular_velocity_gain},{parameter_node.passed_waypoint},{end_time - start_time}"
        )
    finally:
        parameter_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
