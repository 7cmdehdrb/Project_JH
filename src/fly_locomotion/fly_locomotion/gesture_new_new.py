import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *
from enum import Enum

import time
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from rotutils import (
    euler_from_quaternion,
    euler_from_rotation_matrix,
    quaternion_from_euler,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_euler,
    compose_transform,
    decompose_transform,
    invert_transform,
    transform_realsense_to_ros,
)


def pose2matrix(pose: Pose) -> np.ndarray:
    """
    Pose를 4x4 변환 행렬로 변환
    """
    translation_vec = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    rotation_quat = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    rotation_mat = rotation_matrix_from_euler(euler_from_quaternion(rotation_quat))
    T_matrix = compose_transform(translation=translation_vec, rotation=rotation_mat)

    return T_matrix


def frame_change(pose_array: PoseArray) -> PoseArray:
    """
    pose_array의 첫 번째 포즈를 기준으로 상대 포즈 계산
    """
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

        quaternion = quaternion_from_rotation_matrix(relative_matrix[:3, :3])
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


class FingerClass(Enum):
    UNKNOWN = 0
    POINTING = 1


class IntegratedLocomotion(Node):
    def __init__(self):
        super().__init__("integrated_locomotion")

        # 로깅용 토픽 출력 여부
        self.__is_publish = True

        # --- 1. 설정 및 변수 초기화 ---
        self.model = tf.keras.models.load_model(
            "/home/min/7cmdehdrb/fuck_flight/model_train_cls3.h5"
        )

        # 시스템 파라미터
        self.linear_max_threshold = 0.597
        self.angular_max_threshold = 0.255

        self.range_z = np.array([0.0, 3.5])

        # 상태 변수
        self.latest_wrist_pose: Pose = None  # 현재 손목 포즈 (Global)
        self.initial_wrist_pos: Pose = None  # 제스처 시작 시 손목 위치
        self.initial_wrist_rot: R = (
            None  # 제스처 시작 시 손목 회전 (Scipy Rotation 객체)
        )

        self.np_player_pose: np.ndarray = None

        # Waypoint 통과를 위한 시작점
        self.player_pos = np.array([-0.98, -7.83, 2.0])
        self.player_rot = R.from_quat([0.0, 0.0, 0.707, -0.707])

        # 제스처 필터링용 변수
        self.on_gesture_active = False  # 로코모션 로직 활성화 상태
        self.prediction_window = []
        self.last_prediction = FingerClass.UNKNOWN  # 1: Unknown, 0: Pointing

        # --- 2. 통신 설정 ---

        # [Input 1] 스켈레톤 데이터 (제스처 인식용)
        self.sub_l_hand_skeleton_pose = self.create_subscription(
            PoseArray,
            "/l_hand_skeleton_pose",
            self.skeleton_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Input 2] 손목 데이터 (이동 제어용)
        self.sub_wrist_pose_origin = self.create_subscription(
            PoseStamped,
            "/wrist_pose_origin",
            self.wrist_origin_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Input 3] 플레이어 포즈 (절대 좌표계)
        self.sub_player_pose = self.create_subscription(
            PoseStamped,
            "/player_pose",
            self.player_pose_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Output]
        self.pub_player = self.create_publisher(
            PoseStamped, "/player_pose_cmd", qos_profile=qos_profile_system_default
        )  # 미사용중
        self.pub_gesture = self.create_publisher(
            String, "/gesture_recognition", qos_profile=qos_profile_system_default
        )
        self.pub_initial_wrist = self.create_publisher(
            PoseStamped, "/initial_wrist_pose", qos_profile=qos_profile_system_default
        )  # 미사용중

        self.last_time = time.time()

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 상대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def wrist_origin_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def player_pose_callback(self, msg: PoseStamped):
        """플레이어의 절대 좌표 업데이트, 4x4 행렬로 저장(self.player_pose)"""
        position = msg.pose.position
        orientation = msg.pose.orientation

        rot_matrix = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()

        self.np_player_pose = np.array(
            [
                [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], position.x],
                [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], position.y],
                [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], position.z],
                [0, 0, 0, 1],
            ]
        )

    def skeleton_callback(self, msg: PoseArray):
        """
        제스처 인식 로직
        self.last_prediction: 마지막으로 인식된 제스처 상태(Enum)
        """
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
            pred_idx = np.argmax(prediction)

            index_len = len(prediction)  # 2~5

            # Sliding Window
            window_size = 10
            self.prediction_window.append(pred_idx)
            if len(self.prediction_window) > window_size:
                self.prediction_window.pop(0)

            # 빈도 분석 - 윈도우에서 가장 많이 나타난 클래스 선택
            if len(self.prediction_window) >= 5:  # 최소 5개 샘플 필요
                counts = [self.prediction_window.count(i) for i in range(index_len)]

                # 가장 빈도가 높은 클래스 선택
                max_count = max(counts)
                predicted_label = FingerClass(value=counts.index(max_count))

            else:
                predicted_label = self.last_prediction

            # 상태 업데이트
            self.last_prediction = predicted_label

            self.pub_gesture.publish(String(data=str(self.last_prediction.name)))

        else:
            self.get_logger().warning(
                "Model not loaded, cannot perform gesture recognition."
            )

    def run(self):
        """move_next_frame을 호출하는 메인 함수"""
        if self.latest_wrist_pose is None:
            return

        raise NotImplementedError("Locomotion logic not implemented yet.")

    def move_next_frame(self):
        """moveNextFrame: 실제 이동 처리 로직"""

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        print(f"DT: {dt}, HZ: {1.0/dt if dt>0 else 'inf'}")

        if dt > 0.1:
            print("Warning: Large dt detected in move_next_frame:", dt)
            return

        # self.player_pos 업데이트

        # Z축 범위 제한
        self.player_pos[2] = np.clip(
            self.player_pos[2], self.range_z[0], self.range_z[1]
        )


def main(args=None):
    rclpy.init(args=args)

    node = IntegratedLocomotion()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(1000.0)  # 100Hz
        while rclpy.ok():
            node.run()
            r.sleep()

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        th.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
