import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *

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


class IntegratedLocomotion(Node):
    def __init__(self):
        super().__init__("integrated_locomotion")

        # 로깅용 토픽 출력 여부
        self.__is_publish = True

        # --- 1. 설정 및 변수 초기화 ---
        self.model = tf.keras.models.load_model(
            "/home/min/7cmdehdrb/fuck_flight/local_model_fist_gpu.h5"
        )

        # 로코모션 파라미터 (Locomotion.cs에서 가져옴)
        self.translation_threshold_pos = 0.024
        self.translation_threshold_neg = -0.015

        self.rotation_threshold = 0.084

        self.linear_velocity_gain = 4.738
        self.angular_velocity_gain = 0.415
        self.linear_max_threshold = 0.597
        self.angular_max_threshold = 0.255

        # 상태 변수
        self.is_pointing = False  # 제스처 인식 결과
        self.on_gesture_active = False  # 로코모션 로직 활성화 상태

        self.latest_wrist_pose = None  # 현재 손목 포즈 (Global)
        self.initial_wrist_pos = None  # 제스처 시작 시 손목 위치
        self.initial_wrist_rot = None  # 제스처 시작 시 손목 회전 (Scipy Rotation 객체)

        self.delta_q_x_axis = None

        # 가상 플레이어 (Virtual Player) 상태 - (0,0,0)에서 시작
        # self.player_pos = np.array([5.0, 2.0, 2.0])

        # Waypoint 통과
        self.player_pos = np.array([-0.98, -7.83, 2.0])

        # 매니퓰레이터 찾기
        # self.player_pos = np.array([0.53, 1.9, 2.0])

        self.player_rot = R.from_quat(
            [0.0, -0.0, 0.7071068286895752, -0.7071068286895752]
        )
        # 제스처 필터링용 변수
        self.prediction_window = []
        self.last_prediction = 1  # 1: Unknown, 0: Pointing

        # --- 2. 통신 설정 ---

        # [Input 1] 스켈레톤 데이터 (제스처 인식용)
        self.sub_skeleton = self.create_subscription(
            PoseArray,
            "/l_hand_skeleton_pose",
            self.skeleton_callback,
            qos_profile=qos_profile_system_default,
        )

        # # [Input 2] 손목 데이터 (이동 제어용)
        # self.sub_wrist = self.create_subscription(
        #     PoseStamped,
        #     "/wrist_pose",
        #     self.wrist_callback,
        #     qos_profile=qos_profile_system_default,
        # )

        self.sub_wrist_origin = self.create_subscription(
            PoseStamped,
            "/wrist_pose_origin",
            self.wrist_origin_callback,
            qos_profile=qos_profile_system_default,
        )

        self.sub_player_pose = self.create_subscription(
            PoseStamped,
            "/player_pose",
            self.player_pose_callback,
            qos_profile=qos_profile_system_default,
        )

        self.player_pose = None

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
        self.pub_linear_vel = self.create_publisher(
            Float32, "/linear_velocity_cmd", qos_profile=qos_profile_system_default
        )

        # timer_callback
        self.pub_angular_vel = self.create_publisher(
            Point, "/angular_velocity_cmd", qos_profile=qos_profile_system_default
        )

        # timer_callback
        self.pub_initial_wrist = self.create_publisher(
            PoseStamped, "/initial_wrist_pose", qos_profile=qos_profile_system_default
        )

        self.pub_linear_disp = self.create_publisher(
            Float32, "/linear_displacement", qos_profile=qos_profile_system_default
        )
        self.pub_angular_disp = self.create_publisher(
            Vector3, "/angular_displacement", qos_profile=qos_profile_system_default
        )

        self.pub_speed_arrow = self.create_publisher(
            Point, "/speed_arrow_vector", qos_profile=qos_profile_system_default
        )

        self.pub_arrow_activation = self.create_publisher(
            Bool, "/speed_arrow_activation", qos_profile=qos_profile_system_default
        )

        self.last_time = time.time()
        self.dt = 0.01

    def player_pose_callback(self, msg: PoseStamped):
        position = msg.pose.position
        orientation = msg.pose.orientation
        rot_matrix = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()
        self.player_pose = np.array(
            [
                [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], position.x],
                [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], position.y],
                [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], position.z],
                [0, 0, 0, 1],
            ]
        )

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
            pred_idx = np.argmax(prediction)

            print(f"Prediction: {prediction}, Predicted Index: {pred_idx}")

            # Sliding Window
            window_size = 10
            self.prediction_window.append(pred_idx)
            if len(self.prediction_window) > window_size:
                self.prediction_window.pop(0)

            # 빈도 분석
            frequency = self.prediction_window.count(not self.last_prediction)

            predicted_label = self.last_prediction
            if self.last_prediction == 0:  # Pointing
                if frequency >= 3:
                    predicted_label = 1
            else:  # Unknown
                if frequency >= 1:
                    predicted_label = 0

            self.last_prediction = predicted_label

            # 상태 업데이트 (String 발행 대신 변수 저장)
            self.is_pointing = predicted_label == 0
            self.pub_gesture.publish(
                String(data="fist" if self.is_pointing else "unknown")
            )

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def wrist_origin_callback(self, msg: PoseStamped):
        if self.player_pose is not None:
            position = msg.pose.position
            orientation = msg.pose.orientation
            rot_matrix = R.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_matrix()

            wrist_pose = np.array(
                [
                    [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], position.x],
                    [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], position.y],
                    [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], position.z],
                    [0, 0, 0, 1],
                ]
            )

            relative_wrist_pose = np.linalg.inv(self.player_pose) @ wrist_pose

            rela_orientation = R.from_matrix(relative_wrist_pose[:3, :3]).as_quat()

            self.latest_wrist_pose = Pose(
                position=Point(
                    x=relative_wrist_pose[0][3],
                    y=relative_wrist_pose[1][3],
                    z=relative_wrist_pose[2][3],
                ),
                orientation=Quaternion(
                    x=rela_orientation[0],
                    y=rela_orientation[1],
                    z=rela_orientation[2],
                    w=rela_orientation[3],
                ),
            )

    def speed_arrow_vector(self, lin_vel: float):
        magnitude = abs(lin_vel)
        magnitude = abs(lin_vel) * (0.2 / 0.598)
        direction = self.delta_q_x_axis if lin_vel >= 0 else -self.delta_q_x_axis
        return direction * magnitude

    def timer_callback(self):
        """Locomotion.cs의 Update() 및 moveNextFrame() 로직 통합"""
        if self.latest_wrist_pose is None:
            return

        # 1. 상태 관리 (LocomotionControl)
        if self.is_pointing:
            if self.on_gesture_active:
                # 제스처 유지 중 -> 이동 계산
                lin_vel = self.calculate_translation_vel()
                ang_vel_vec = self.calculate_rotation_vel()
                self.move_next_frame(lin_vel, ang_vel_vec)

                self.pub_arrow_activation.publish(Bool(data=True))

                speed_arrow_vec = self.speed_arrow_vector(lin_vel)
                self.pub_speed_arrow.publish(
                    Point(
                        x=speed_arrow_vec[0], y=speed_arrow_vec[1], z=speed_arrow_vec[2]
                    )
                )

                # 로깅용 토픽 발행
                if self.__is_publish:
                    self.pub_linear_vel.publish(Float32(data=lin_vel))
                    self.pub_angular_vel.publish(
                        Point(x=ang_vel_vec[0], y=ang_vel_vec[1], z=ang_vel_vec[2])
                    )

            else:
                # 제스처 시작 순간 -> 초기값 저장
                self.on_gesture_active = True
                self.initial_wrist_pos = np.array(
                    [
                        self.latest_wrist_pose.position.x,
                        self.latest_wrist_pose.position.y,
                        self.latest_wrist_pose.position.z,
                    ]
                )
                self.initial_wrist_rot = R.from_quat(
                    [
                        self.latest_wrist_pose.orientation.x,
                        self.latest_wrist_pose.orientation.y,
                        self.latest_wrist_pose.orientation.z,
                        self.latest_wrist_pose.orientation.w,
                    ]
                )

        else:
            # 제스처 풀림
            self.on_gesture_active = False
            self.pub_speed_arrow.publish(Point(x=0.0, y=0.0, z=0.0))
            self.pub_arrow_activation.publish(Bool(data=False))

        # 2. 결과 Publish (PoseStamped)
        qx, qy, qz, qw = self.player_rot.as_quat()
        out_msg = PoseStamped(
            header=Header(
                # stamp=self.get_clock().now().to_msg(),
                stamp=rclpy.time.Time(seconds=0).to_msg(),
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
            if self.initial_wrist_rot is not None:
                initial_wrist_quat = self.initial_wrist_rot.as_quat()

                initial_wrist_msg = PoseStamped(
                    header=Header(
                        stamp=self.get_clock().now().to_msg(),
                        # stamp=rclpy.time.Time(seconds=0).to_msg(),
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

    def calculate_translation_vel(self):
        """calculateTranslation + linearVelocityMapping"""
        current_pos = np.array(
            [
                self.latest_wrist_pose.position.x,
                self.latest_wrist_pose.position.y,
                self.latest_wrist_pose.position.z,
            ]
        )

        # C# 코드: delta_z = relativeWristPosition.y - initialWristPosition.y
        # Unity Y (Up) -> ROS Z (Up). 손의 높이 변화를 속도로 매핑
        # delta_h = current_pos[2] - self.initial_wrist_pos[2]
        delta_h = current_pos[0] - self.initial_wrist_pos[0]

        # 로깅용 토픽 발행
        if self.__is_publish:
            self.pub_linear_disp.publish(Float32(data=delta_h))

        lin_vel = 0.0
        # Deadzone check
        if self.translation_threshold_neg < delta_h < self.translation_threshold_pos:
            lin_vel = 0.0
        elif delta_h > 0:
            lin_vel = (
                delta_h - self.translation_threshold_pos
            ) * self.linear_velocity_gain
        else:
            lin_vel = (
                delta_h - self.translation_threshold_neg
            ) * self.linear_velocity_gain

        # Max Speed Clamp
        if abs(lin_vel) > self.linear_max_threshold:
            lin_vel = self.linear_max_threshold * np.sign(lin_vel)

        return lin_vel

    def calculate_rotation_vel(self):
        """calculateRotation + angularVelocityMapping"""
        current_rot = R.from_quat(
            [
                self.latest_wrist_pose.orientation.x,
                self.latest_wrist_pose.orientation.y,
                self.latest_wrist_pose.orientation.z,
                self.latest_wrist_pose.orientation.w,
            ]
        )

        # C# Logic: delta_q = current * Inverse(initial)
        # Scipy Logic: 순서 주의 (Global frame 기준 회전 차이)
        delta_q = current_rot * self.initial_wrist_rot.inv()

        # delta_q는 이미 Rotation 객체이므로 as_matrix()로 변환
        delta_q_matrix = delta_q.as_matrix()
        self.delta_q_x_axis = delta_q_matrix[:, 0].tolist()

        norm_delta_q_x = np.linalg.norm(self.delta_q_x_axis)
        if norm_delta_q_x < 1e-6:
            return np.zeros(3)
        self.delta_q_x_axis /= norm_delta_q_x

        # Convert delta_q to Euler angles (roll, pitch, yaw)
        euler_angles = delta_q.as_euler("xyz", degrees=False)

        if self.__is_publish:
            self.pub_angular_disp.publish(
                Vector3(x=euler_angles[0], y=euler_angles[1], z=euler_angles[2])
            )

        # To Angle-Axis
        rot_vec = delta_q.as_rotvec()  # 크기가 각도(rad), 방향이 축
        delta_angle = np.linalg.norm(rot_vec)

        if delta_angle < 1e-6:  # 0 나누기 방지
            return np.zeros(3)

        delta_axis = rot_vec / delta_angle

        # Threshold check
        if abs(delta_angle) < self.rotation_threshold:
            return np.zeros(3)

        # Gain 적용
        ang_vel_mag = (
            delta_angle - self.rotation_threshold
        ) * self.angular_velocity_gain

        # Max Speed Clamp
        if ang_vel_mag > self.angular_max_threshold:
            ang_vel_mag = self.angular_max_threshold

        return delta_axis * ang_vel_mag

    def move_next_frame(self, lin_vel, ang_vel_vec):
        """moveNextFrame: 적분하여 위치/회전 업데이트"""

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        print(f"DT: {dt}, HZ: {1.0/dt if dt>0 else 'inf'}")

        if dt > 0.3:
            print("Warning: Large dt detected in move_next_frame:", dt)
            return  # 너무 큰 dt 방지

        # 1. Linear Movement (플레이어가 바라보는 방향으로 전진)
        player_rot_matrix = self.player_rot.as_matrix()
        forward_vec = player_rot_matrix[:, 0]  # X축 (Forward)

        displacement = forward_vec * lin_vel * dt  # self.dt
        self.player_pos += displacement

        if self.player_pos[2] > 3.5:
            self.player_pos[2] = 3.5

        delta_rotvec = ang_vel_vec * dt

        if np.linalg.norm(delta_rotvec) > 1e-6:
            # 축-각 벡터에서 직접 회전 객체 생성
            delta_rot = R.from_rotvec(delta_rotvec)

            # 회전 누적: player_rot = player_rot * delta_rot
            self.player_rot = self.player_rot * delta_rot


def main(args=None):
    rclpy.init(args=args)

    node = IntegratedLocomotion()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    try:
        r = node.create_rate(100.0)  # 100Hz
        while rclpy.ok():
            node.timer_callback()
            r.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
