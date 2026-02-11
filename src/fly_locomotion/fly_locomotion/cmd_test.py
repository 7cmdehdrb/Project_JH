import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *

import time
import numpy as np
from scipy.spatial.transform import Rotation as R


class CmdNode(Node):
    def __init__(self):
        super().__init__("cmd_node")

        self.__wrist_origin = PoseStamped()
        self.__initial_wrist_origin = PoseStamped()
        self.__disable_wrist_origin = PoseStamped(
            header=Header(),
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=-999.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

        self.__previous_player_pos: PoseStamped = None
        self.__player_pos_sub = self.create_subscription(
            PoseStamped,
            "/player_pose",
            callback=self.__player_pos_callback,
            qos_profile=qos_profile_system_default,
        )

        self.__wrist_sub = self.create_subscription(
            PoseStamped,
            "/wrist_pose_origin",
            callback=self.__wrist_callback,
            qos_profile=qos_profile_system_default,
        )

        self.__point = False
        self.__point_sub = self.create_subscription(
            String,
            "/gesture_recognition",
            callback=self.__point_callback,
            qos_profile=qos_profile_system_default,
        )

        self.__wristpub = self.create_publisher(
            PoseStamped,
            "/wrist_pose_origin_cmd",
            qos_profile=qos_profile_system_default,
        )

    def __player_pos_callback(self, msg: PoseStamped):
        # Calculate position delta if previous position exists
        if self.__previous_player_pos is not None:
            delta_x = msg.pose.position.x - self.__previous_player_pos.pose.position.x
            delta_y = msg.pose.position.y - self.__previous_player_pos.pose.position.y
            delta_z = msg.pose.position.z - self.__previous_player_pos.pose.position.z

            # Apply delta to __initial_wrist_origin
            self.__initial_wrist_origin.pose.position.x += delta_x
            self.__initial_wrist_origin.pose.position.y += delta_y
            self.__initial_wrist_origin.pose.position.z += delta_z

        # Update current and previous player positions
        self.__previous_player_pos = PoseStamped(
            header=msg.header,
            pose=Pose(
                position=Point(
                    x=msg.pose.position.x, y=msg.pose.position.y, z=msg.pose.position.z
                ),
                orientation=Quaternion(
                    x=msg.pose.orientation.x,
                    y=msg.pose.orientation.y,
                    z=msg.pose.orientation.z,
                    w=msg.pose.orientation.w,
                ),
            ),
        )

    def __point_callback(self, msg: String):
        current_data = msg.data == "pointing"

        if self.__point is False and current_data is True:
            # The case when the gesture just changed to pointing
            print("Pointing gesture started")
            self.__initial_wrist_origin = self.__wrist_origin

        self.__point = current_data

    def __wrist_callback(self, msg: PoseStamped):
        temp = msg

        original_rotation = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )

        # Create 90 degree rotation around z-axis
        z_rotation = R.from_euler("y", np.deg2rad(-90), degrees=False)
        original_rot = R.from_quat(original_rotation)
        rotated = z_rotation * original_rot
        rotated_quat = rotated.as_quat()

        # Apply rotated quaternion to temp
        temp.pose.orientation = Quaternion(
            x=rotated_quat[0], y=rotated_quat[1], z=rotated_quat[2], w=rotated_quat[3]
        )

        self.__wrist_origin = temp

    def run(self):
        # Case 1: No pointing gesture detected
        if self.__point is False:
            cmd_msg = self.__disable_wrist_origin

        # Case 2: Pointing gesture detected
        else:
            cmd_msg = self.__initial_wrist_origin

        self.__wristpub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)

    node = CmdNode()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(100.0)
        while rclpy.ok():
            node.run()
            r.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
