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


class WaypointPosReset(Node):
    def __init__(self):
        super().__init__("waypoint_pos_reset")

        self.__waypoint_pos_reset_pub = self.create_publisher(
            PoseStamped,
            "/waypoint_pos_cmd",
            qos_profile=qos_profile_system_default,
        )
        self.__reset = False

        self.__timer = self.create_timer(1.0, self.publish_reset)

    def publish_reset(self):
        msg = (
            PoseStamped(
                header=Header(
                    stamp=self.get_clock().now().to_msg(),
                    frame_id="world",
                ),
                pose=Pose(
                    position=Point(x=0.0, y=0.0, z=-999.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
            if self.__reset
            else PoseStamped(
                header=Header(
                    stamp=self.get_clock().now().to_msg(),
                    frame_id="world",
                ),
                pose=Pose(
                    position=Point(x=0.0, y=0.0, z=-1.56),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
        )

        self.__waypoint_pos_reset_pub.publish(msg)


def main():
    rclpy.init(args=None)

    node = WaypointPosReset()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
