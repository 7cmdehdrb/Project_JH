#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String, Float32MultiArray

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from transforms3d.quaternions import quat2mat, mat2quat

# --- GPU 메모리 오류 방지 설정 ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit=false"

# GPU 환경이 불안정할 경우 CPU 모드로 강제 전환
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 초기화 에러: {e}")


# --- 좌표 변환 유틸리티 ---
def pose2matrix(pose):
    q = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    rotation_mat = quat2mat(q)
    T_matrix = np.eye(4)
    T_matrix[0:3, 0:3] = rotation_mat
    T_matrix[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T_matrix


class HandPoseClassifier(Node):
    def __init__(self):
        super().__init__("hand_pose_classifier")

        # 1. 파일 경로 설정 (v2 모델과 스케일러)
        base_path = "/home/jinju/Downloads/locomotion_technique_skeleton_version"
        model_path = os.path.join(base_path, "hand_pose_model_v2.h5")
        scaler_path = os.path.join(base_path, "hand_pose_model_v2_scaler.pkl")

        # 2. 모델 및 스케일러 로드
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.get_logger().info("✅ v2 모델 및 스케일러 로드 성공!")
        except Exception as e:
            self.get_logger().error(f"❌ 로드 실패: {e}")
            exit()

        # 3. 구독자 및 발행자 설정
        self.sub = self.create_subscription(
            PoseArray, "/l_hand_skeleton_pose", self.callback, 10
        )

        # 요청하신 두 개의 토픽 발행자
        self.pub_label = self.create_publisher(String, "/gesture", 10)
        self.pub_prob = self.create_publisher(Float32MultiArray, "/gesture_array", 10)

        self.class_names = [
            "Translation",
            "Rotation",
            "Unknown",
        ]
        self.get_logger().info("🚀 실시간 듀얼 토픽 추론기 시작")

    def frame_change(self, msg_poses):
        """학습 시와 동일한 손목 기준 168차원 피처 추출"""
        features = []
        if not msg_poses or len(msg_poses) < 1:
            return None

        # 0번 관절(Wrist) 기준 상대 변환
        ref_mat = pose2matrix(msg_poses[0])
        inv_ref = np.linalg.inv(ref_mat)

        for pose in msg_poses:
            rel_matrix = np.dot(inv_ref, pose2matrix(pose))
            q = mat2quat(rel_matrix[0:3, 0:3])  # [w, x, y, z]

            # 피처 순서: [x, y, z, qx, qy, qz, qw]
            features.extend(
                [
                    rel_matrix[0, 3],
                    rel_matrix[1, 3],
                    rel_matrix[2, 3],
                    q[1],
                    q[2],
                    q[3],
                    q[0],
                ]
            )
        return features

    def callback(self, msg):
        # 1. 특징 변환 (168차원 추출)
        features = self.frame_change(msg.poses)
        if features is None or len(features) != 168:
            return

        # 2. 정규화 (스케일러 적용)
        input_data = np.array([features])
        input_scaled = self.scaler.transform(input_data)

        # 3. 모델 추론
        prediction = self.model.predict(input_scaled, verbose=0)  # shape: (1, 3)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]

        # 4. 결과 결정 (신뢰도 0.9 기준 필터링)
        label_msg = String()

        if confidence >= 0.6:
            # 신뢰도가 0.9 이상일 때만 모델의 예측 결과 그대로 사용
            label_msg.data = self.class_names[class_idx]
            self.get_logger().info(
                f"Gesture: {label_msg.data} (Conf: {confidence:.2f})"
            )
        else:
            # 신뢰도가 0.9 미만이면 "Unknown (Nothing)"으로 강제 변경
            # self.class_names[2]가 "Unknown (Nothing)"인 경우
            label_msg.data = self.class_names[2]
            self.get_logger().warn(
                f"Low Confidence ({confidence:.2f}) -> Forcing Unknown"
            )

        # 5. 결과 발행 (듀얼 전송)
        # (1) String 라벨 발행: /gesture
        self.pub_label.publish(label_msg)

        # (2) Float32MultiArray 확률 발행: /gesture_array (원본 확률값 유지)
        prob_msg = Float32MultiArray()
        prob_msg.data = prediction[0].tolist()
        self.pub_prob.publish(prob_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
