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
        self.translation_threshold_neg = -0.01

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
        self.last_prediction = 1  # 1: Unknown, 0: Pointing

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
        self.pub_linear_vel = self.create_publisher(
            Vector3, "/linear_velocity_cmd", qos_profile=qos_profile_system_default
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
            Vector3, "/linear_displacement", qos_profile=qos_profile_system_default
        )
        self.pub_angular_disp = self.create_publisher(
            Vector3, "/angular_displacement", qos_profile=qos_profile_system_default
        )

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
            pred_idx = np.argmax(prediction)

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
                String(data="pointing" if self.is_pointing else "unknown")
            )

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def timer_callback(self):
        """Locomotion.cs의 Update() 및 moveNextFrame() 로직 통합"""
        if self.latest_wrist_pose is None:
            return

        # 1. 상태 관리 (LocomotionControl)
        if self.is_pointing:
            if self.on_gesture_active:
                # 제스처 유지 중 -> 이동 계산
                lin_vel_vec = self.calculate_translation_vel()
                ang_vel_vec = self.calculate_rotation_vel()
                self.move_next_frame(lin_vel_vec, ang_vel_vec)

                # 로깅용 토픽 발행
                if self.__is_publish:
                    self.pub_linear_vel.publish(
                        Vector3(x=lin_vel_vec[0], y=lin_vel_vec[1], z=lin_vel_vec[2])
                    )
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
            if self.initial_wrist_rot is not None:
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

    def _linear_velocity_mapping(self, delta):
        """단일 축에 대한 선형 속도 매핑 (데드존 + 게인 + 클램프)"""
        vel = 0.0
        if self.translation_threshold_neg < delta < self.translation_threshold_pos:
            vel = 0.0
        elif delta > 0:
            vel = (delta - self.translation_threshold_pos) * self.linear_velocity_gain
        else:
            vel = (delta - self.translation_threshold_neg) * self.linear_velocity_gain

        # Max Speed Clamp
        if abs(vel) > self.linear_max_threshold:
            vel = self.linear_max_threshold * np.sign(vel)

        return vel

    def calculate_translation_vel(self):
        """손의 x,y,z 변위를 각각 선형 속도로 매핑하여 3D 속도 벡터 반환"""
        current_pos = np.array(
            [
                self.latest_wrist_pose.position.x,
                self.latest_wrist_pose.position.y,
                self.latest_wrist_pose.position.z,
            ]
        )

        delta = current_pos - self.initial_wrist_pos  # [dx, dy, dz]

        # 로깅용 토픽 발행
        if self.__is_publish:
            self.pub_linear_disp.publish(
                Vector3(x=float(delta[0]), y=float(delta[1]), z=float(delta[2]))
            )

        # 각 축에 대해 독립적으로 선형 속도 매핑
        lin_vel_vec = np.array(
            [
                self._linear_velocity_mapping(delta[0]),
                self._linear_velocity_mapping(delta[1]),
                self._linear_velocity_mapping(delta[2]),
            ]
        )

        return lin_vel_vec

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

    def move_next_frame(self, lin_vel_vec, ang_vel_vec):
        """moveNextFrame: x,y,z 각 축 독립 속도와 각속도로 위치/회전 업데이트"""

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        print(f"DT: {dt}, HZ: {1.0/dt if dt>0 else 'inf'}")

        if dt > 0.1:
            print("Warning: Large dt detected in move_next_frame:", dt)
            return  # 너무 큰 dt 방지

        # 1. Linear Movement (x,y,z 각 축 독립 속도로 직접 이동)
        displacement = lin_vel_vec * dt
        self.player_pos += displacement

        if self.player_pos[2] > 3.5:
            self.player_pos[2] = 3.5

        # 2. Angular Movement
        delta_rotvec = ang_vel_vec * dt

        if np.linalg.norm(delta_rotvec) > 1e-6:
            delta_rot = R.from_rotvec(delta_rotvec)
            self.player_rot = self.player_rot * delta_rot


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
            r.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
