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

# TensorFlow
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

# Scipy for transformation
from scipy.spatial.transform import Rotation as R

def pose2matrix(pose):
    rotation = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
    translation = [pose.position.x, pose.position.y, pose.position.z]
    T_matrix = [[rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
                [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
                [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
                [0, 0, 0, 1]]
    return T_matrix

def frame_change(pose_array):
        relative_pose_array = PoseArray()
        relative_pose_array.header.frame_id = "wrist"
        reference = pose_array.poses[0]
        reference_matrix = pose2matrix(reference)
        for pose in pose_array.poses:
            pose_matrix = pose2matrix(pose)
            relative_matrix = np.dot(np.linalg.inv(reference_matrix), pose_matrix)
            relative_pose = Pose()
            relative_pose.position.x = relative_matrix[0][3]
            relative_pose.position.y = relative_matrix[1][3]
            relative_pose.position.z = relative_matrix[2][3]
            quaternion = R.from_matrix(relative_matrix[:3,:3]).as_quat()
            relative_pose.orientation.x = quaternion[0]
            relative_pose.orientation.y = quaternion[1]
            relative_pose.orientation.z = quaternion[2]
            relative_pose.orientation.w = quaternion[3]
            relative_pose_array.poses.append(relative_pose)
        return relative_pose_array

class Gesture(Node):
    def __init__(self):
        super().__init__("test")
        self.model = tf.keras.models.load_model("/home/hoon/hand_skeleton/skeleton_dataset/model/model_train_241201.h5")

        self.create_subscription(
            msg_type=PoseArray,
            topic='l_hand_skeleton_pose',
            callback=self.skeleton_callback,
            qos_profile=QoSProfile(depth=10)
        )

        self.pub = self.create_publisher(
            msg_type=String, topic="gesture_past", qos_profile=QoSProfile(depth=10)
        )

    #     self.timer = self.create_timer(1.0, callback=self.timer_callback)  # 1.0 Hz

    # def timer_callback(self):
    #     pass

    def skeleton_callback(self, msg: PoseArray):
        poses = frame_change(msg)
        skeleton_data = []
        for pose in poses.poses:
            skeleton_data.append(pose.position.x)
            skeleton_data.append(pose.position.y)
            skeleton_data.append(pose.position.z)
            skeleton_data.append(pose.orientation.x)
            skeleton_data.append(pose.orientation.y)
            skeleton_data.append(pose.orientation.z)
            skeleton_data.append(pose.orientation.w)
        skeleton_data = np.array([skeleton_data]).astype(float)
        # print(skeleton_data.shape)
        prediction = self.model.predict(skeleton_data)
        predicted_label = np.argmax(prediction)
        print(predicted_label, prediction, end=' ')
        print(prediction[0].tolist())
        # self.pub_array.publish(label_array)
        if predicted_label == 0:
            label = "Translation"
            self.pub.publish(String(data=label))
        elif predicted_label == 1:
            label = "Rotation"
            self.pub.publish(String(data=label))
        else:
            label = "Unknown"
            self.pub.publish(String(data=label))

def main():
    rclpy.init(args=None)

    node = Gesture()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
