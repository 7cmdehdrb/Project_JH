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

# GPU 설정 (필요시 활성화)
# tf.config.set_visible_devices([], "GPU")


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


class FingerClass(Enum):
    TWO = 0
    ONE = 1
    UNKNOWN = 2


class IntegratedLocomotion(Node):
    def __init__(self):
        super().__init__("integrated_locomotion")

        # 로깅용 토픽 출력 여부
        self.__is_publish = True

        # --- 1. 설정 및 변수 초기화 ---
        self.model = tf.keras.models.load_model(
            "/home/min/7cmdehdrb/fuck_flight/model_train_cls3.h5"
        )

        # 로코모션 파라미터 (Locomotion.cs에서 가져옴)
        self.translation_threshold_pos = 0.02
        self.translation_threshold_neg = -0.01

        self.rotation_threshold = 0.01

        self.linear_velocity_gain = 4.44
        self.angular_velocity_gain = 0.417
        self.linear_max_threshold = 0.597
        self.angular_max_threshold = 0.255

        # 상태 변수
        self.is_pointing = FingerClass.UNKNOWN  # 제스처 인식 결과
        self.on_gesture_active = False  # 로코모션 로직 활성화 상태

        self.latest_wrist_pose = None  # 현재 손목 포즈 (Global)
        self.initial_wrist_pos = None  # 제스처 시작 시 손목 위치
        self.initial_wrist_rot = None  # 제스처 시작 시 손목 회전 (Scipy Rotation 객체)

        # 가상 플레이어 (Virtual Player) 상태 - (0,0,0)에서 시작
        # self.player_pos = np.array([5.0, 2.0, 2.0])

        # Waypoint 통과
        self.player_pos = np.array([-0.98, -7.83, 2.0])
        self.player_rot = R.from_quat(
            [0.0, -0.0, 0.7071068286895752, -0.7071068286895752]
        )

        # 매니퓰레이터 찾기
        # self.player_pos = np.array([0.53, 1.9, 2.0])
        # self.player_rot = R.from_quat([0.0, 0.0, 0.707, -0.707])  # Identity

        # 제스처 필터링용 변수
        self.prediction_window = []
        self.last_prediction = FingerClass.UNKNOWN  # 1: Unknown, 0: Pointing

        # --- 2. 통신 설정 ---

        # [Input 1] 스켈레톤 데이터 (제스처 인식용)
        self.sub_skeleton = self.create_subscription(
            PoseArray,
            "/l_hand_skeleton_pose",
            self.skeleton_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Input 2] 손목 데이터 (이동 제어용)
        self.sub_wrist = self.create_subscription(
            PoseStamped,
            "/wrist_pose",
            self.wrist_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Output]
        # timer_callback
        self.pub_player = self.create_publisher(
            PoseStamped, "/player_pose_cmd", qos_profile=qos_profile_system_default
        )
        # skeleton_callback
        self.pub_gesture = self.create_publisher(
            String, "/gesture_recognition", qos_profile=qos_profile_system_default
        )

        # timer_callback
        # self.pub_linear_vel = self.create_publisher(
        #     Float32, "/linear_velocity_cmd", qos_profile=qos_profile_system_default
        # )

        # timer_callback
        # self.pub_angular_vel = self.create_publisher(
        #     Point, "/angular_velocity_cmd", qos_profile=qos_profile_system_default
        # )

        # timer_callback
        self.pub_initial_wrist = self.create_publisher(
            PoseStamped, "/initial_wrist_pose", qos_profile=qos_profile_system_default
        )

        # self.pub_linear_disp = self.create_publisher(
        #     Float32, "/linear_displacement", qos_profile=qos_profile_system_default
        # )
        # self.pub_angular_disp = self.create_publisher(
        #     Vector3, "/angular_displacement", qos_profile=qos_profile_system_default
        # )

        self.last_time = time.time()
        self.dt = 0.01

    def skeleton_callback(self, msg: PoseArray):
        """제스처 인식 로직"""
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
            # [[0.01455264 0.83905846 0.14638886]] -> [[주먹, 포인팅, NULL]]
            pred_idx = np.argmax(prediction)

            # Sliding Window
            window_size = 10
            self.prediction_window.append(pred_idx)
            if len(self.prediction_window) > window_size:
                self.prediction_window.pop(0)

            # 빈도 분석 - 윈도우에서 가장 많이 나타난 클래스 선택
            if len(self.prediction_window) >= 5:  # 최소 5개 샘플 필요
                count_0 = self.prediction_window.count(0)
                count_1 = self.prediction_window.count(1)
                count_2 = self.prediction_window.count(2)

                # 가장 빈도가 높은 클래스 선택
                max_count = max(count_0, count_1, count_2)
                if count_0 == max_count:
                    predicted_label = FingerClass.UNKNOWN
                elif count_1 == max_count:
                    predicted_label = FingerClass.ONE
                else:  # count_2 == max_count
                    predicted_label = FingerClass.TWO
            else:
                predicted_label = self.last_prediction

            self.last_prediction = predicted_label

            # 상태 업데이트
            self.is_pointing = predicted_label

            # 제스처 문자열 매핑
            gesture_names = {
                FingerClass.UNKNOWN: "unknown",
                FingerClass.ONE: "one",
                FingerClass.TWO: "two",
            }
            self.pub_gesture.publish(
                String(data=gesture_names.get(predicted_label, "unknown"))
            )

        else:
            self.get_logger().warning(
                "Model not loaded, cannot perform gesture recognition."
            )

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def timer_callback(self):
        """Locomotion.cs의 Update() 및 moveNextFrame() 로직 통합"""
        if self.latest_wrist_pose is None:
            return

        # 1. 제스처에 따른 선형 속도 결정
        speed_map = {
            FingerClass.UNKNOWN: 0.0,
            FingerClass.ONE: 0.596 * (1.2 / 3.2),  # 0.597 * 1.0,  # 1.2
            FingerClass.TWO: 0.597,  # 0.597 * 2.0,  # , 3.2
        }
        lin_vel = speed_map.get(self.is_pointing, 0.0)

        # 24m -> 30초

        # 2. 상태 관리: 제스처 활성화 시 initial_wrist_rot 저장
        if self.is_pointing != FingerClass.UNKNOWN:
            if not self.on_gesture_active:
                # 제스처 시작 순간 -> 초기값 저장
                self.on_gesture_active = True
                self.initial_wrist_rot = R.from_quat(
                    [
                        self.latest_wrist_pose.orientation.x,
                        self.latest_wrist_pose.orientation.y,
                        self.latest_wrist_pose.orientation.z,
                        self.latest_wrist_pose.orientation.w,
                    ]
                )

            # 3. 이동 방향: initial_wrist_rot을 원점(1,0,0)으로 삼고,
            #    현재 손목 회전의 상대 변화를 방향 벡터로 변환
            current_rot = R.from_quat(
                [
                    self.latest_wrist_pose.orientation.x,
                    self.latest_wrist_pose.orientation.y,
                    self.latest_wrist_pose.orientation.z,
                    self.latest_wrist_pose.orientation.w,
                ]
            )
            delta_rot = current_rot * self.initial_wrist_rot.inv()

            # delta_rot을 각속도(rotvec)로 사용 -> 플레이어 회전에 누적
            ang_vel_vec = delta_rot.as_rotvec()

            delta_euler = delta_rot.as_euler("xyz", degrees=True)
            print(
                f"Delta Euler: Roll: {delta_euler[0]:.2f}, Pitch: {delta_euler[1]:.2f}, Yaw: {delta_euler[2]:.2f}"
            )

            # 4. 이동 적용: 플레이어의 forward 방향으로 전진 + 포인팅 방향으로 회전
            if lin_vel > 0.0:
                self.move_next_frame(lin_vel, ang_vel_vec)
        else:
            # 제스처 풀림
            self.on_gesture_active = False

        # 2. 결과 Publish (PoseStamped)
        qx, qy, qz, qw = self.player_rot.as_quat()
        out_msg = PoseStamped(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id="map",
            ),
            pose=Pose(
                position=Point(
                    x=self.player_pos[0],
                    y=self.player_pos[1],
                    z=self.player_pos[2],
                ),
                orientation=Quaternion(
                    x=qx,
                    y=qy,
                    z=qz,
                    w=qw,
                ),
            ),
        )
        self.pub_player.publish(out_msg)

        # 로깅용 토픽 발행
        if self.__is_publish:
            if (
                self.initial_wrist_rot is not None
                and self.initial_wrist_pos is not None
            ):
                initial_wrist_quat = self.initial_wrist_rot.as_quat()

                initial_wrist_msg = PoseStamped(
                    header=Header(
                        stamp=self.get_clock().now().to_msg(),
                        frame_id="map",
                    ),
                    pose=Pose(
                        position=Point(
                            x=self.initial_wrist_pos[0],
                            y=self.initial_wrist_pos[1],
                            z=self.initial_wrist_pos[2],
                        ),
                        orientation=Quaternion(
                            x=initial_wrist_quat[0],
                            y=initial_wrist_quat[1],
                            z=initial_wrist_quat[2],
                            w=initial_wrist_quat[3],
                        ),
                    ),
                )

                self.pub_initial_wrist.publish(initial_wrist_msg)

    def move_next_frame(self, lin_vel, ang_vel_vec):
        """moveNextFrame: forward 방향으로 전진 + 포인팅 방향으로 회전"""

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        print(f"DT: {dt}, HZ: {1.0/dt if dt>0 else 'inf'}")

        if dt > 0.1:
            print("Warning: Large dt detected in move_next_frame:", dt)
            return

        # 1. 회전: 포인팅 방향(delta_rot)을 각속도로 사용하여 플레이어 회전 누적
        delta_rotvec = ang_vel_vec * dt
        if np.linalg.norm(delta_rotvec) > 1e-6:
            delta_rot = R.from_rotvec(delta_rotvec)
            self.player_rot = self.player_rot * delta_rot

        # 2. 전진: 플레이어가 현재 바라보는 방향(X축)으로 이동
        forward_vec = self.player_rot.apply([1.0, 0.0, 0.0])
        displacement = forward_vec * lin_vel * dt
        self.player_pos += displacement

        if self.player_pos[2] > 3.5:
            self.player_pos[2] = 3.5


def main(args=None):
    rclpy.init(args=args)

    node = IntegratedLocomotion()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(1000.0)  # 100Hz
        while rclpy.ok():
            node.timer_callback()
            pass
            r.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
