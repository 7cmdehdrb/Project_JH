import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *

import time
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

def pose2matrix(pose):
    rotation = R.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    ).as_matrix()
    translation = [pose.position.x, pose.position.y, pose.position.z]
    T_matrix = np.eye(4, dtype=np.float64)
    T_matrix[:3, :3] = rotation
    T_matrix[:3, 3] = translation
    return T_matrix


def frame_change(pose_array: PoseArray) -> PoseArray:
    relative_pose_array = PoseArray(
        header=Header(
            stamp=pose_array.header.stamp,
            frame_id="wrist",
        ),
        poses=[],
    )

    if not pose_array.poses:
        return relative_pose_array

    reference = pose_array.poses[0]
    reference_matrix = pose2matrix(reference)
    inv_reference_matrix = np.linalg.inv(reference_matrix)

    for pose in pose_array.poses:
        pose_matrix = pose2matrix(pose)
        relative_matrix = np.dot(inv_reference_matrix, pose_matrix)

        quaternion = R.from_matrix(relative_matrix[:3, :3]).as_quat()
        relative_pose = Pose(
            position=Point(
                x=relative_matrix[0][3],
                y=relative_matrix[1][3],
                z=relative_matrix[2][3],
            ),
            orientation=Quaternion(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            ),
        )

        relative_pose_array.poses.append(relative_pose)

    return relative_pose_array

class RealtimeModelTest(Node):
    def __init__(self):
        super().__init__('realtime_model_test')

        qos_profile = QoSProfile(depth=10)

        self.subscription = self.create_subscription(
            PoseArray,
            '/l_hand_skeleton_pose',
            self.pose_array_callback,
            qos_profile)

        self.model = tf.keras.models.load_model(
            "/home/hoon/hand_skeleton/fly_locomotion/model/model_mlp_tuned_20260211_180741.h5"
        )

        self.get_logger().info('Realtime Model Test Node has been started.')

    def pose_array_callback(self, msg):
        poses = frame_change(msg)
        skeleton_data = []
        for pose in poses.poses:
            pose: Pose
            skeleton_data.extend(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )

        skeleton_data = np.array([skeleton_data]).astype(float)

        # 모델 추론
        if hasattr(self, "model"):
            prediction = self.model.predict(skeleton_data, verbose=0)
            
            print(f"Raw model output: {prediction}")
            pred_idx = np.argmax(prediction)
            print(f"Predicted class index: {pred_idx}")



            # # Sliding Window
            # window_size = 10
            # self.prediction_window.append(pred_idx)
            # if len(self.prediction_window) > window_size:
            #     self.prediction_window.pop(0)

            # # 빈도 분석
            # frequency = self.prediction_window.count(not self.last_prediction)

            # predicted_label = self.last_prediction
            # if self.last_prediction == 0:  # Pointing
            #     if frequency >= 3:
            #         predicted_label = 1
            # else:  # Unknown
            #     if frequency >= 1:
            #         predicted_label = 0

            # self.last_prediction = predicted_label

            # # 상태 업데이트 (String 발행 대신 변수 저장)
            # self.is_pointing = predicted_label == 0
            # self.pub_gesture.publish(
            #     String(data="pointing" if self.is_pointing else "unknown")
            # )


def main(args=None):
    rclpy.init(args=args)

    realtime_model_test = RealtimeModelTest()

    try:
        rclpy.spin(realtime_model_test)
    except KeyboardInterrupt:
        pass
    finally:
        realtime_model_test.file.close()
        realtime_model_test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()