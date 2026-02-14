import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *

import time
import numpy as np


class SkeletonSaver(Node):
    def __init__(self):
        super().__init__('skeleton_saver')

        qos_profile = QoSProfile(depth=10)

        self.subscription = self.create_subscription(
            PoseArray,
            '/l_hand_skeleton_pose',
            self.pose_array_callback,
            qos_profile)
        self.subscription  # prevent unused variable warning

        # self.label_subscription = self.create_subscription(
        #     String,
        #     '/hand_skeleton/label',
        #     self.label_callback,
        #     qos_profile)
        # self.label_subscription  # prevent unused variable warning

        self.file = open('raw_skeleton_data.txt', 'a')

        self.current_label = 'unknown'
        self.get_logger().info('Skeleton Saver Node has been started.')

    def pose_array_callback(self, msg):
        pose_list = []
        for pose in msg.poses:
            pose_list.append([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ])
        pose_array = np.array(pose_list).flatten()
        data_line = ','.join(map(str, pose_array.tolist())) + ',' + self.current_label + '\n'
        self.file.write(data_line)
        self.get_logger().info(f'Saved skeleton data with label: {self.current_label}')

    def label_callback(self, msg):
        self.current_label = msg.data
        self.get_logger().info(f'Current label set to: {self.current_label}')

def main(args=None):
    rclpy.init(args=args)

    skeleton_saver = SkeletonSaver()

    try:
        rclpy.spin(skeleton_saver)
    except KeyboardInterrupt:
        pass

    skeleton_saver.destroy_node()
    rclpy.shutdown()
    skeleton_saver.file.close()

if __name__ == '__main__':
    main()