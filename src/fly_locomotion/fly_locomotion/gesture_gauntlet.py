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

        # 데드존 파라미터
        self.translation_threshold_pos = 0.024  # 선형 + 데드존
        self.translation_threshold_neg = -0.01  # 선형 - 데드존
        self.rotation_threshold = 0.084  # 회전 데드존

        self.linear_velocity_gain = 4.738
        self.angular_velocity_gain = 0.415
        self.linear_max_threshold = 0.597  # 최대 속도 클리핑
        self.angular_max_threshold = 0.255  # 최대 각속도 클리핑

        # 손목 최대 회전각 처리 파라미터. 인체 기구학에 근거한 값임으로 수정시 유의
        # 주석의 기준은 왼손 손등이 좌측을 향한 기준임
        """
        TO. 정재훈
        사실 이거 맞는 값인지 몰라. Gemini가 말하는 값 그대로 쓴거야
        """
        self.max_flexion = np.deg2rad(70.0)  # 굴곡. 우회전
        self.max_extension = np.deg2rad(70.0)  # 신전. 좌회전
        self.max_radial_deviation = np.deg2rad(15.0)  # 요측 편위. 상승
        self.max_ulnar_deviation = np.deg2rad(30.0)  # 척측 편위. 하강

        # 상태 변수
        self.is_pointing = False  # 제스처 인식 결과
        self.on_gesture_active = False  # 로코모션 로직 활성화 상태

        self.latest_wrist_pose = None  # 현재 손목 포즈 (Global)
        self.initial_wrist_pos = None  # 제스처 시작 시 손목 위치
        self.initial_wrist_rot = None  # 제스처 시작 시 손목 회전 (Scipy Rotation 객체)

        # Initial Player Pose 정의
        self.player_pose = None
        self.player_pos = np.array([-0.98, -7.83, 2.0])
        self.player_rot = R.from_quat(
            [0.0, -0.0, 0.7071068286895752, -0.7071068286895752]
        )  # 중요: 시작 점이 y축 방향을 보도록 설계 되어 있음

        self.init_player_pos = self.player_pos.copy()
        self.init_player_rot = self.player_rot

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
        )  # 스켈레톤 데이터, 로컬

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

        self.__ball_cnt_sub = self.create_subscription(
            Int32,
            "/ball_count",
            self.ball_count_callback,
            qos_profile=qos_profile_system_default,
        )
        self.cnt: int = 0

        # [Input 2] 손목 데이터 (이동 제어용)
        # self.sub_wrist = self.create_subscription(
        #     PoseStamped,
        #     "/wrist_pose",
        #     self.wrist_callback,
        #     qos_profile=qos_profile_system_default,
        # )  # 손목 좌표, 글로벌

        # [Output]
        self.pub_player = self.create_publisher(
            PoseStamped, "/player_pose_cmd", qos_profile=qos_profile_system_default
        )  # 플레이어 포즈 명령 발행

        self.pub_gesture = self.create_publisher(
            String, "/gesture_recognition", qos_profile=qos_profile_system_default
        )  # 제스처 인식 결과 발행

        self.pub_linear_vel = self.create_publisher(
            Vector3, "/linear_velocity_cmd", qos_profile=qos_profile_system_default
        )  # 로깅용 선형 속도 발행

        self.pub_angular_vel = self.create_publisher(
            Point, "/angular_velocity_cmd", qos_profile=qos_profile_system_default
        )  # 로깅용 각속도 발행

        self.pub_initial_wrist = self.create_publisher(
            PoseStamped, "/initial_wrist_pose", qos_profile=qos_profile_system_default
        )  # 로깅용 초기 손목 포즈 발행

        self.pub_linear_disp = self.create_publisher(
            Vector3, "/linear_displacement", qos_profile=qos_profile_system_default
        )  # 로깅용 선형 변위 발행

        self.pub_angular_disp = self.create_publisher(
            Vector3, "/angular_displacement", qos_profile=qos_profile_system_default
        )  # 로깅용 각 변위 발행

        self.last_time = time.time()

    def ball_count_callback(self, msg: Int32):
        """Ball Count 업데이트 및 수집 로직"""

        if self.cnt != msg.data:
            self.player_pos = self.init_player_pos.copy()
            self.player_rot = self.init_player_rot

        self.cnt = msg.data

    def skeleton_callback(self, msg: PoseArray):
        """제스처 인식 로직"""

        # STEP 1: 스켈레톤 데이터 전처리 (프레임 변경)
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
            # STEP 2: 모델 예측

            prediction = self.model.predict(skeleton_data, verbose=0)
            pred_idx = np.argmax(prediction)

            # STEP 3: 후처리 - 이동 윈도우 및 빈도 분석
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

            # STEP 4: 결과 처리 및 발행
            self.last_prediction = predicted_label
            self.is_pointing = predicted_label == 0
            self.pub_gesture.publish(
                String(data="pointing" if self.is_pointing else "unknown")
            )

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

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        self.latest_wrist_pose = msg.pose

    def timer_callback(self):
        """Locomotion.cs의 Update() 및 moveNextFrame() 로직 통합"""
        if self.latest_wrist_pose is None:
            return

        # 1. 제스처 상태에 따른 이동/회전 계산
        if self.is_pointing:
            # 제스처 인식됨
            if self.on_gesture_active:

                # STEP 1: 선형 속도 계산
                # 각 축방향 변위에, 속도 게인 곱해서 선형 속도 벡터 산출
                lin_vel_vec = self.calculate_translation_vel()

                # STEP 2: 각속도 계산
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

        """
        NOTICE: delta는 해당 축에 대한 변위. 
        self.translation_threshold_pos = 0.024
        self.translation_threshold_neg = -0.01 처럼 정의 되어 있음
        
        즉, 각 축의 선형 변위가 -0.01 ~ 0.024 사이일 때는 속도 0.
        양수 방향으로 0.024 이상 이동 시, (delta - 0.024) * gain 만큼 속도 증가.
        음수 방향으로 -0.01 이하 이동 시, (delta + 0.01) * gain 만큼 속도 증가. 
        최대 속도는 self.linear_max_threshold 로 클램프.
        """

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
        # NOTICE: 각 축에 대한 선형 변화랑을 입력
        lin_vel_vec = np.array(
            [
                self._linear_velocity_mapping(delta[0]),
                self._linear_velocity_mapping(delta[1]),
                self._linear_velocity_mapping(delta[2]),
            ]
        )

        return lin_vel_vec

    def calculate_rotation_vel(self):
        # 1. 현재 손목의 쿼터니언 생성
        current_rot = R.from_quat(
            [
                self.latest_wrist_pose.orientation.x,
                self.latest_wrist_pose.orientation.y,
                self.latest_wrist_pose.orientation.z,
                self.latest_wrist_pose.orientation.w,
            ]
        )

        # 2. 초기 각도와의 차이(Delta) 계산
        delta_q = current_rot * self.initial_wrist_rot.inv()

        # 3. 오일러 각도로 변환 (roll, pitch, yaw). 각각 float
        d_roll, d_pitch, d_yaw = delta_q.as_euler("xyz", degrees=False)

        # 계산된 각도 Publish
        if self.__is_publish:
            self.pub_angular_disp.publish(Vector3(x=d_roll, y=d_pitch, z=d_yaw))

        # ------------------------------------------------------------------
        # 새로운 로직: 각 축별 물리적 한계에 따른 정규화 매핑
        # ------------------------------------------------------------------

        # 출력 벡터 초기화
        vel_x = 0.0  # Roll: 항상 0.0으로 고정
        vel_y = 0.0  # Pitch: 상/하 (Radial/Ulnar)
        vel_z = 0.0  # Yaw: 좌/우 (Flexion/Extension)

        target_speed = self.angular_max_threshold  # 0.255

        """
        TO. 정재훈
        PITCH 움직임이 반대로 이동할 경우, 부호를 변경할 것.
        """
        # [매핑 1] Pitch (Y축) 출력 계산: 요측/척측 편위
        # 비율 계산: 현재각도 / 최대각도 (1.0을 넘지 않도록 클램핑)
        if abs(d_yaw) < self.rotation_threshold:
            # 데드존 처리
            vel_y = 0.0
        elif d_yaw > 0.0:
            # 상승 로직.
            ratio = min(abs(d_yaw) / self.max_radial_deviation, 1.0)
            vel_y = ratio * target_speed  # 양수(+) 방향
        else:
            # 하강 로직
            ratio = min(abs(d_yaw) / self.max_ulnar_deviation, 1.0)
            vel_y = -1.0 * ratio * target_speed  # 음수(-) 방향

        """
        TO. 정재훈
        YAW 움직임이 반대로 이동할 경우, 부호를 변경할 것.
        """
        # [매핑 2] Yaw (Z축) 출력 계산: 굴곡/신전 (Flexion/Extension)
        if abs(d_pitch) < self.rotation_threshold:
            # 데드존 처리
            vel_z = 0.0
        elif d_pitch > 0.0:
            ratio = min(abs(d_pitch) / self.max_flexion, 1.0)
            vel_z = 1.0 * ratio * target_speed
        else:
            ratio = min(abs(d_pitch) / self.max_extension, 1.0)
            vel_z = -1.0 * ratio * target_speed

        """
        TO. 정재훈
        이 부분은 로컬 좌표계라서, Unity 쪽 좌표계에 맞게 축을 재배치 해야 할 수도 있음.
        지금 졸려서 머리가 안돌아가는데, 단순하게 순서를 변경하는게 아니라, 축에 맞춰서 threshold나 신전/굴곡/편위를 다시 설정해야 할 수도 있음.
        ㅠㅠ
        """

        final_vel = np.array([vel_x, vel_z, vel_y])

        """
        최종 속도 벡터 조합
        각 축이 독립적이라, 전체 회전이 self.angular_max_threshold 를 넘길 수 있기 때문에, 전체 벡터 크기에 대한 클램프 적용
        만약, 각 축에 threshold를 적용하는 것에서 끝나고 싶다면, 위에서 바로 return final_vel 해도 됨.
        """
        current_magnitude = np.linalg.norm(final_vel)

        # 만약 현재 속도 벡터의 크기가 최대 제한을 초과한다면
        if current_magnitude > self.angular_max_threshold:
            # 방향은 유지한 채 크기만 제한값으로 줄임 (Scaling)
            scale_factor = self.angular_max_threshold / current_magnitude
            final_vel = final_vel * scale_factor

        return final_vel

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
        global_lin_vel_vec = self.player_rot.as_matrix() @ lin_vel_vec
        displacement = global_lin_vel_vec * dt
        self.player_pos += displacement

        # if self.player_pos[2] > 3.5:
        #     self.player_pos[2] = 3.5

        # 2. Angular Movement
        delta_rotvec = ang_vel_vec * dt
        d_roll, d_pitch, d_yaw = delta_rotvec
        delta_rotvec = np.array([d_roll, d_pitch, d_yaw])

        if np.linalg.norm(delta_rotvec) > 1e-6:
            delta_rot = R.from_rotvec(delta_rotvec)
            self.player_rot = self.player_rot * delta_rot

        """
        TO. 정재훈
        roll을 0.0으로 고정했으나, Unity 쪽애서 좌표계가 달라서 다른 축 회전이 고정될 수 있음.
        그럴 경우, pitch나 yaw를 죽이고, roll만 유지하는 방식으로 수정 필요.
        """
        # roll, pitch, yaw = self.player_rot.as_euler("xyz", degrees=False)
        # self.player_rot = R.from_euler(
        #     "xyz",
        #     [0.0, pitch, yaw],
        #     degrees=False,
        # )


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
