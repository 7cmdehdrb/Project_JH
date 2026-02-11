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

# Python
import numpy as np
import os
from datetime import datetime, timezone
from loguru import logger
import struct
import socket



class SocketClient:
    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port
        self.client_socket = None

    def connect(self):
        """Connect to server"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

    def send_float_array(self, float_array):
        """Send array of 7 floats to server"""
        if len(float_array) != 7:
            raise ValueError("Array must contain exactly 7 floats")

        # Ensure float32 type
        float_array = np.array(float_array, dtype=np.float32)

        # Pack 7 floats into bytes
        data = struct.pack("7f", *float_array)

        # Send to server
        self.client_socket.sendall(data)

    def close(self):
        """Close connection"""
        if self.client_socket:
            self.client_socket.close()
            print("Connection closed")


class DataSaver(Node):
    def __init__(self):
        super().__init__("data_saver")

        self.__socket_client = SocketClient(host="220.149.89.165", port=5555)
        self.__socket_client.connect()

        self.subscription_waypoint = self.create_subscription(
            PoseArray,
            "/waypoint_array",
            self.pose_array_callback,
            qos_profile_system_default,
        )

        self.subscription_player_pose = self.create_subscription(
            PoseStamped,
            "/player_pose",
            self.player_pose_callback,
            qos_profile_system_default,
        )    

        time = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs(f"data/{time}", exist_ok=True)
        self.waypoint_file = open(f"data/{time}/waypoints.txt", "w")
        self.waypoint_flag = False

        # logger.add(f"data/{time}/loguru_{time}.csv", format="{message}")
        # logger.info("server_time,position_x,position_y,position_z,orientation_x,orientation_y,orientation_z,orientation_w")

    def shutdown(self):
        self.__socket_client.close()

    def pose_array_callback(self, msg: PoseArray):
        if self.waypoint_flag is False:
            if len(msg.poses) > 0:
                for i, pose in enumerate(msg.poses):
                    pose: Pose
                    server_time = datetime.now(timezone.utc).isoformat()
                    data = (f"{server_time},{i},{pose.position.x},{pose.position.y},{pose.position.z},{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}")
                    self.waypoint_file.write(data + "\n")
                self.waypoint_flag = True
                self.waypoint_file.close()

    def player_pose_callback(self, msg: PoseStamped):
        server_time = datetime.now(timezone.utc).isoformat()

        float_array = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        self.__socket_client.send_float_array(float_array)
        print(f"Sent player pose at {server_time}")

        # logger.info(f"{server_time},{msg.pose.position.x},{msg.pose.position.y},{msg.pose.position.z},{msg.pose.orientation.x},{msg.pose.orientation.y},{msg.pose.orientation.z},{msg.pose.orientation.w}")

def main(args=None):
    rclpy.init(args=args)

    import threading
    data_saver = DataSaver()

    thread = threading.Thread(target=rclpy.spin, args=(data_saver,), daemon=True)
    thread.start()

    while rclpy.ok():
        pass

    data_saver.shutdown()
    data_saver.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()