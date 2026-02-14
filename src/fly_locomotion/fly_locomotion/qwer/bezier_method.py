import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *
from enum import Enum
from typing import List, Tuple, Optional, Any
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


import numpy as np
from scipy.spatial.transform import Rotation as R


def rot_from_fwd_keep_world_up(
    fwd_world: np.ndarray,
    world_up: np.ndarray,
    prev_right_world: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[R, np.ndarray]:
    """
    fwd_world 방향을 유지하면서 roll이 없도록(로컬 up을 world_up에 맞추도록) 회전을 재구성.
    반환: (Rotation, right_world)  -- right_world는 다음 프레임 연속성 유지에 사용.

    로컬 축 정의(중요):
      - 로컬 +X = forward
      - 로컬 +Y = right
      - 로컬 +Z = up
    이 축 정의가 다르면 행렬 구성 부분을 바꿔야 한다.
    """

    f = np.asarray(fwd_world, dtype=np.float64)
    u = np.asarray(world_up, dtype=np.float64)

    fn = np.linalg.norm(f)
    un = np.linalg.norm(u)
    if fn < eps or un < eps:
        # 입력이 망가진 경우: identity
        return R.identity(), np.array([0.0, 1.0, 0.0], dtype=np.float64)

    f /= fn
    u /= un

    # right = up x forward
    r = np.cross(u, f)
    rn = np.linalg.norm(r)

    if rn < eps:
        # forward가 world_up과 거의 평행 -> right가 정의 불가
        # 이전 right가 있으면 그걸로 right를 만들고, 없으면 보조축 사용
        if prev_right_world is not None:
            r = prev_right_world.copy()
            # right를 forward에 직교하도록 정규화(Gram-Schmidt)
            r = r - f * np.dot(r, f)
            rn = np.linalg.norm(r)
        if rn < eps:
            # prev_right도 못 쓰면 보조축으로 생성
            aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(aux, f)) > 0.9:
                aux = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            r = np.cross(aux, f)
            rn = np.linalg.norm(r)

    r /= max(rn, eps)

    # up = forward x right  (정확히 직교화)
    up = np.cross(f, r)
    up /= max(np.linalg.norm(up), eps)

    # 회전행렬 구성: columns = [forward, right, up]
    # 로컬축(+X fwd, +Y right, +Z up)을 월드로 보내는 행렬
    Rm = np.column_stack([f, r, up])
    rot = R.from_matrix(Rm)

    return rot, r


import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def build_upright_blended(
    raw_rot: R,
    world_up: np.ndarray,
    local_fwd: np.ndarray,
    local_up: np.ndarray,
    prev_right: np.ndarray | None,
    vertical_start=0.85,  # |dot(fwd, world_up)|가 이 이상이면 "수직 근처" 취급
    vertical_end=0.98,
):
    """
    - local_fwd/local_up: 당신 로컬축 정의(예: +Z forward면 [0,0,1])
    - world_up: 월드 up
    - prev_right: 연속성 유지용
    """

    world_up = normalize(world_up)

    fwd = normalize(raw_rot.apply(local_fwd))
    raw_up = normalize(raw_rot.apply(local_up))

    # 수직 정도 측정: 0이면 수평, 1이면 완전 수직
    v = abs(np.dot(fwd, world_up))

    # v가 커질수록(world_up과 평행) world_up 고정을 약하게 하고 raw_up을 더 신뢰
    # t=0 -> world_up 사용, t=1 -> raw_up 사용
    if v <= vertical_start:
        t = 0.0
    elif v >= vertical_end:
        t = 1.0
    else:
        t = (v - vertical_start) / (vertical_end - vertical_start)

    # 블렌드 up 후보
    up_ref = normalize((1.0 - t) * world_up + t * raw_up)

    # right, up 재구성
    right = np.cross(up_ref, fwd)
    rn = np.linalg.norm(right)

    if rn < 1e-6:
        # still degenerate -> prev_right로 보정
        if prev_right is not None:
            right = prev_right - fwd * np.dot(prev_right, fwd)
            right = normalize(right)
        else:
            # fallback
            aux = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(aux, fwd)) > 0.9:
                aux = np.array([0.0, 1.0, 0.0])
            right = normalize(np.cross(aux, fwd))

    up = normalize(np.cross(fwd, right))

    # 로컬축 매핑: columns = [local_fwd, local_right, local_up]를 월드로 보낸다 가정
    Rm = np.column_stack([fwd, right, up])
    return R.from_matrix(Rm), right


def remove_axis_twist_xyzw(q_xyzw: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Swing-Twist decomposition에서 axis에 대한 twist를 제거하고 swing만 반환.
    입력/출력 쿼터니언은 [x, y, z, w] (SciPy 포맷), 반드시 unit quaternion에 가깝다고 가정.
    axis는 월드 좌표계 기준 축(예: [1,0,0]).
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)

    # normalize
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / qn

    an = np.linalg.norm(axis)
    if an < 1e-12:
        return q.copy()
    axis = axis / an

    v = q[:3]
    w = q[3]

    # v를 axis 방향으로 투영 -> twist의 벡터부 후보
    proj = axis * np.dot(v, axis)
    twist = np.array([proj[0], proj[1], proj[2], w], dtype=np.float64)

    tn = np.linalg.norm(twist)
    # pitch가 ±90도 근처 등으로 w≈0이고 proj≈0이면 twist 정의가 붕괴할 수 있음 -> twist=identity 처리
    if tn < 1e-12:
        twist = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        twist /= tn

    # swing = q * inv(twist)  (q = swing * twist 라는 가정에 대응)
    # inv(unit quat) = conjugate
    t_conj = np.array([-twist[0], -twist[1], -twist[2], twist[3]], dtype=np.float64)

    # quat multiply (q * t_conj)
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = t_conj

    swing = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )

    # normalize + sign 안정화(연속성)
    swing /= max(np.linalg.norm(swing), 1e-12)
    if swing[3] < 0:  # w가 음수로 튀는 것 방지(동일회전의 부호 반전)
        swing = -swing

    return swing


def pose2matrix(pose: Pose) -> np.ndarray:
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
    POINTING = 0
    UNKNOWN = 1


class LowPassFilter:
    def __init__(self, cutoff_freq, ts):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        self.pre_out = 0.0
        self.tau = self.calc_filter_coef()

    def calc_filter_coef(self):
        w_cut = 2 * np.pi * self.cutoff_freq
        return 1 / w_cut

    def filter(self, data):
        out = (self.tau * self.pre_out + self.ts * data) / (self.tau + self.ts)
        self.pre_out = out
        return out


class AverageFilter:
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.data_window = []

    def filter(self, data: float) -> float:
        if self.window_size == 1:
            return data

        self.data_window.append(data)
        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)
        return sum(self.data_window) / len(self.data_window)


class PointAverageFilter:
    def __init__(self, window_size: int):
        self.xf = AverageFilter(window_size)
        self.yf = AverageFilter(window_size)
        self.zf = AverageFilter(window_size)

    def filter(self, pose: Pose) -> Pose:
        assert isinstance(pose, Pose), "Input must be a geometry_msgs/Pose"

        x, y, z = pose.position.x, pose.position.y, pose.position.z

        x_filt = self.xf.filter(x)
        y_filt = self.yf.filter(y)
        z_filt = self.zf.filter(z)

        return Pose(
            position=Point(x=x_filt, y=y_filt, z=z_filt),
            orientation=pose.orientation,  # Orientation is not filtered
        )


class Bezier3D:
    """
    Quadratic Bezier curve in 3D:
        B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    where P0=start, P2=end, and P1 is a gain-adjusted control point.

    Gain behavior (stable and intuitive):
        P1_eff = midpoint + gain * (control - midpoint)
    so:
        gain = 0   -> straight-line-ish (control collapses to midpoint)
        gain = 1   -> original control point
        gain > 1   -> stronger bend
        gain < 0   -> bends to the opposite side
    """

    def __init__(
        self,
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 0.0, 0.0]),
        control=np.array([0.0, 0.0, 0.0]),
        gain=1.0,
        dtype=np.float64,
    ):
        self._start = self._as_vec3(start, dtype)
        self._end = self._as_vec3(end, dtype)
        self._control = self._as_vec3(control, dtype)
        self._gain = float(gain)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value: List | Tuple | np.ndarray):
        self._start = self._as_vec3(value, dtype=np.float64)

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value: List | Tuple | np.ndarray):
        self._end = self._as_vec3(value, dtype=np.float64)

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, value: List | Tuple | np.ndarray):
        self._control = self._as_vec3(value, dtype=np.float64)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = float(value)

    @staticmethod
    def parse_curve(data: np.ndarray, stamp: rclpy.time.Time) -> PoseArray:
        msg = PoseArray(
            header=Header(
                stamp=stamp,
                frame_id="world",
            ),
            poses=[],
        )

        list_data = data.T.tolist()
        for li in list_data:
            if len(li) != 3:
                raise ValueError("Each point must be length-3.")

            pose = Pose(
                position=Point(x=li[0], y=li[1], z=li[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            msg.poses.append(pose)

        return msg

    def start_normal(self):
        """
        Returns the unit normal vector at the start point (t = 0)
        using the Frenet frame definition.
        """

        P0 = self.start
        P1 = self.control
        P2 = self.end

        # First derivative at t=0
        d1 = 2.0 * (P1 - P0)

        norm_d1 = np.linalg.norm(d1)
        if norm_d1 < 1e-12:
            raise ValueError("Tangent vector at start is zero; normal is undefined.")

        T = d1 / norm_d1  # unit tangent

        # Second derivative (constant for quadratic Bezier)
        d2 = 2.0 * (P2 - 2.0 * P1 + P0)

        # Remove tangential component
        N = d2 - np.dot(d2, T) * T

        norm_N = np.linalg.norm(N)
        if norm_N < 1e-12:
            raise ValueError("Normal vector at start is undefined (zero curvature).")

        return N / norm_N

    def _as_vec3(self, x, dtype):
        a = np.asarray(x, dtype=dtype).reshape(-1)
        if a.shape[0] != 3:
            raise ValueError(f"Point must be length-3, got shape {a.shape}.")
        return a

    def effective_control(self):
        mid = 0.5 * (self.start + self.end)
        return mid + self.gain * (self.control - mid)

    def sample(self, n, include_endpoints=True):
        """
        Returns:
            pts: (3, n) numpy array of sampled points along the curve.
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        # Parameter t
        if include_endpoints:
            t = np.linspace(0.0, 1.0, n, dtype=self.start.dtype)
        else:
            # avoids exactly 0 and 1
            t = np.linspace(0.0, 1.0, n + 2, dtype=self.start.dtype)[1:-1]

        P0 = self.start[:, None]  # (3,1)
        P2 = self.end[:, None]  # (3,1)
        P1 = self.control[:, None]  # (3,1)

        u = 1.0 - t  # (n,)
        uu = (u * u)[None, :]  # (1,n)
        ut2 = (2.0 * u * t)[None, :]  # (1,n)
        tt = (t * t)[None, :]  # (1,n)

        pts = uu * P0 + ut2 * P1 + tt * P2  # (3,n)
        return pts


class IntegratedLocomotion(Node):
    def __init__(self):
        super().__init__("integrated_locomotion")

        # 로깅용 토픽 출력 여부
        self.__is_publish = True

        self.bezier_curve: Bezier3D = Bezier3D()

        # --- 1. 설정 및 변수 초기화 ---
        self.model = tf.keras.models.load_model(
            "/home/min/7cmdehdrb/fuck_flight/local_model_gpu.h5"
        )

        # 시스템 파라미터
        self.dead_zone_angle = np.deg2rad(5.0)  # 10도 데드존
        self.rotation_gain = 1.0
        self.linear_max_threshold = 0.597
        self.angular_max_threshold = 0.255

        self.range_z = np.array([-999.0, 999.0])

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

        # Bezier Curve용 변수 선언
        self.hmd_pose: Pose = None
        self.hmd_pos: np.ndarray = None
        self.hmd_rot = R.identity()

        self.init_player_pos = self.player_pos.copy()
        self.init_player_rot = self.player_rot

        self.index_fingertip_pose: Pose = None
        self.bezier_norm = np.array([0.0, 0.0, 1.0])

        # HMD, Index Finger Tip Pose 필터
        self.hmd_pose_filter = PointAverageFilter(window_size=1)
        self.index_finger_tip_pose_filter = PointAverageFilter(window_size=1)

        # __init__ 같은 곳에
        self.prev_right_world = np.array(
            [0.0, 1.0, 0.0], dtype=np.float64
        )  # 초기값 아무거나
        self.world_up = np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )  # z-up 가정(필요시 변경)

        # --- 2. 통신 설정 ---

        # [Input 1] 스켈레톤 데이터 (제스처 인식용)
        self.sub_l_hand_skeleton_pose = self.create_subscription(
            PoseArray,
            "/l_hand_skeleton_pose",
            self.skeleton_callback,
            qos_profile=qos_profile_system_default,
        )

        self.sub_wrist_pose = self.create_subscription(
            PoseStamped,
            "/wrist_pose",
            self.wrist_callback,
            qos_profile=qos_profile_system_default,
        )

        # [Input 2] 손목 데이터 (이동 제어용)
        self.sub_wrist_pose_origin = self.create_subscription(
            PoseStamped,
            "/wrist_pose_origin",
            self.wrist_origin_callback,
            qos_profile=qos_profile_system_default,
        )

        self.sub_hmd_pose = self.create_subscription(
            PoseStamped,
            "/hmd_pose",
            self.hmd_pose_callback,
            qos_profile=qos_profile_system_default,
        )

        self.sub_index_finger_tip_pose = self.create_subscription(
            PoseStamped,
            "/index_fingertip_pose",
            self.index_fingertip_pose_callback,
            qos_profile=qos_profile_system_default,
        )

        self.pub_bezier_curve = self.create_publisher(
            PoseArray, "/bezier_curve", qos_profile=qos_profile_system_default
        )

        self.__ball_cnt_sub = self.create_subscription(
            Int32,
            "/ball_count",
            self.ball_count_callback,
            qos_profile=qos_profile_system_default,
        )
        self.cnt: int = 0

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

    def ball_count_callback(self, msg: Int32):
        """Ball Count 업데이트 및 수집 로직"""

        if self.cnt != msg.data:
            self.player_pos = self.init_player_pos.copy()
            self.player_rot = self.init_player_rot

        self.cnt = msg.data

    def wrist_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 상대 좌표 업데이트"""
        # self.latest_wrist_pose = msg.pose
        pass

    def wrist_origin_callback(self, msg: PoseStamped):
        """이동/회전 계산을 위한 손목의 절대 좌표 업데이트"""
        # print("Wrist Origin Callback Triggered")
        self.latest_wrist_pose = msg.pose

    def hmd_pose_callback(self, msg: PoseStamped):
        # print("HMD Pose Callback Triggered")
        pose = self.hmd_pose_filter.filter(msg.pose)

        self.hmd_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.hmd_rot = R.from_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )

        self.hmd_pose = pose

    def index_fingertip_pose_callback(self, msg: PoseStamped):
        # print("Index Fingertip Pose Callback Triggered")
        pose = self.index_finger_tip_pose_filter.filter(msg.pose)
        self.index_fingertip_pose = pose

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
                    predicted_label = FingerClass(value=0)
                elif count_1 == max_count:
                    predicted_label = FingerClass(value=1)
            else:
                predicted_label = self.last_prediction

            # 상태 업데이트
            self.last_prediction = predicted_label

            self.pub_gesture.publish(String(data=str(self.last_prediction.name)))

        else:
            self.get_logger().warning(
                "Model not loaded, cannot perform gesture recognition."
            )

    def make_bezier_curve(self) -> Optional[np.ndarray]:
        if self.hmd_pose is None or self.index_fingertip_pose is None:
            self.get_logger().warning(
                "HMD pose or Index fingertip pose is None, cannot create Bezier curve."
            )
            return None

        # SET Bezier Curve Points
        self.bezier_curve.start = np.array(
            [
                self.hmd_pose.position.x,
                self.hmd_pose.position.y,
                self.hmd_pose.position.z,
            ]
        )

        self.bezier_curve.end = np.array(
            [
                self.index_fingertip_pose.position.x,
                self.index_fingertip_pose.position.y,
                self.index_fingertip_pose.position.z,
            ]
        )

        last_vec: np.ndarray = self.bezier_curve.end - self.bezier_curve.start

        T_hmd = pose2matrix(self.hmd_pose)
        T_C = np.array(
            [
                [1.0, 0.0, 0.0, 0.05],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T_world_C = T_hmd @ T_C

        self.bezier_curve.control = self.hmd_pos + np.array(
            [self.hmd_rot.apply([0.3, 0.0, 0.0])]
        )
        self.bezier_curve.gain = 1.0  # 제어점의 영향력 조절

        bezier_curve: np.ndarray = self.bezier_curve.sample(
            n=30, include_endpoints=True
        )

        bezier_curve_T = bezier_curve.T
        normal_vec: np.ndarray = bezier_curve_T[1] - bezier_curve_T[0]

        normal_vec = np.linalg.inv(T_hmd[:3, :3]) @ normal_vec

        msg = Bezier3D.parse_curve(
            data=bezier_curve, stamp=self.get_clock().now().to_msg()
        )
        self.pub_bezier_curve.publish(msg)

        max_dist = 0.5
        min_dist = 0.2
        norm_last_vec = np.clip(
            (np.linalg.norm(last_vec) - min_dist) / (max_dist - min_dist), 0.0, 1.0
        )

        return normal_vec, norm_last_vec, last_vec

    def run(self):
        """move_next_frame을 호출하는 메인 함수"""

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if self.latest_wrist_pose is None:
            self.get_logger().warning(
                "Latest wrist pose is None, cannot run locomotion."
            )
            return

        speed_map = {
            FingerClass.UNKNOWN: 0.0,
            FingerClass.POINTING: self.linear_max_threshold,
        }
        lin_vel = speed_map.get(self.last_prediction, 0.0)

        if self.last_prediction != FingerClass.UNKNOWN:
            # normal_vec: 이동해야 할 상대 벡터
            normal_vec, dist, last_dist = self.make_bezier_curve()
            lin_vel *= dist  # 거리 비례 속도 조절

            # print(f"Linear Velocity Command: {lin_vel:.3f} m/s")

            # Exception 처리
            if normal_vec is None:
                self.get_logger().warning(
                    "Failed to create Bezier curve normal vector."
                )
                return None

            # -----------------------------
            # 여기부터: player_rot이 normal_vec을 바라보게 회전
            # -----------------------------
            eps = 1e-12

            dst = np.asarray(normal_vec, dtype=float).reshape(3)
            dst = self.hmd_rot.apply(dst)
            player_vec = self.hmd_rot.apply([1.0, 0.0, 0.0])

            # 데드존 처리: 목표 벡터가 기본 전방과 거의 같으면 그냥 기본 전방 사용
            last_dst = np.asarray(last_dist, dtype=float).reshape(3)
            angle_between = np.arccos(
                np.clip(
                    np.dot(player_vec, last_dst)
                    / (np.linalg.norm(player_vec) * np.linalg.norm(last_dst)),
                    -1.0,
                    1.0,
                )
            )

            # if angle_between < self.dead_zone_angle:
            #     dst = player_vec

            # 업데이트
            # self.player_pos = self.player_pos + dst * lin_vel * dt

            # ================================================

            dst_norm = np.linalg.norm(dst)
            if dst_norm < eps:
                self.get_logger().warning(
                    "normal_vec is near zero; cannot orient player."
                )
                return None
            dst /= dst_norm  # 목표 전방(단위벡터)

            # 현재 플레이어 전방: 로컬 +X가 전방이라고 가정
            cur = self.hmd_rot.apply([1.0, 0.0, 0.0])
            cur_norm = np.linalg.norm(cur)
            if cur_norm < eps:
                self.get_logger().warning(
                    "Current forward vector is near zero; cannot orient player."
                )
                return None
            cur /= cur_norm

            dot = float(np.clip(np.dot(cur, dst), -1.0, 1.0))

            # gain: 0이면 회전 안 함, 1이면 원래 회전, 2면 2배 회전(과회전)
            gain = float(self.rotation_gain)  # 예: 클래스 멤버로 두거나 파라미터로 받기
            gain = max(0.0, gain)  # 음수는 보통 의미 없어서 막음(원하면 허용해도 됨)

            # 이미 거의 같은 방향이면 회전 생략
            if dot > 1.0 - 1e-9:
                delta_rot = R.identity()

            # 정반대 방향이면 축이 무한히 많아서 임의의 직교축으로 180도 회전
            elif dot < -1.0 + 1e-9:
                tmp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(cur, tmp)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])

                axis = np.cross(cur, tmp)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < eps:
                    self.get_logger().warning(
                        "Failed to find a valid axis for 180-degree rotation."
                    )
                    return None
                axis /= axis_norm

                # 180도 회전도 gain으로 선형 스케일
                delta_rot = R.from_rotvec((np.pi * gain) * axis)

            else:
                # 일반 케이스: 축 = cur x dst, 각 = acos(dot)
                axis = np.cross(cur, dst)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < eps:
                    delta_rot = R.identity()
                else:
                    axis /= axis_norm
                    angle = np.arccos(dot)

                    # angle을 gain만큼 선형 스케일
                    delta_rot = R.from_rotvec((angle * gain) * axis)

            # roll, pitch, yaw = delta_rot.as_euler("xyz", degrees=False)
            # 다시 Rotation으로 구성
            # delta_rot = R.from_euler("xyz", [roll, 0.0, yaw], degrees=False)

            # -----------------------------

            print(f"DT: {dt}, HZ: {1.0/dt if dt>0 else 'inf'}")
            if dt > 0.1:
                print("Warning: Large dt detected in move_next_frame:", dt)
                return

            next_hmd_pos = self.hmd_pos + dst * lin_vel * dt  # 100Hz 가정
            next_hmd_rot = delta_rot * self.hmd_rot
            next_hmd_rot_matrix = next_hmd_rot.as_matrix()

            next_hmd_transform = compose_transform(
                translation=next_hmd_pos,
                rotation=next_hmd_rot_matrix,
            )

            current_hmd_transform = compose_transform(
                translation=self.hmd_pos,
                rotation=self.hmd_rot.as_matrix(),
            )

            current_player_transform = compose_transform(
                translation=self.player_pos,
                rotation=self.player_rot.as_matrix(),
            )

            next_player_transform = (
                next_hmd_transform
                @ np.linalg.inv(current_hmd_transform)
                @ current_player_transform
            )

            # 업데이트
            self.player_pos = next_player_transform[:3, 3]

            # raw_rot = R.from_matrix(next_player_transform[:3, :3])

            # # 좌표계에 맞게 지정 (예시)
            # world_up = np.array([0.0, 0.0, 1.0])
            # local_fwd = np.array([1.0, 0.0, 0.0])  # 네가 확정한 forward 축
            # local_up = np.array([0.0, 0.0, 1.0])  # 네가 확정한 up 축

            # self.player_rot, self.prev_right_world = build_upright_blended(
            #     raw_rot=raw_rot,
            #     world_up=world_up,
            #     local_fwd=local_fwd,
            #     local_up=local_up,
            #     prev_right=getattr(self, "prev_right_world", None),
            # )

            self.player_rot = R.from_matrix(next_player_transform[:3, :3])

            roll, pitch, yaw = self.player_rot.as_euler("xyz", degrees=False)
            # print(roll, pitch, yaw)

            self.player_rot = R.from_euler("xyz", [0.0, pitch, yaw], degrees=False)
            # self.player_rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)

            # Z축 범위 제한
            self.player_pos[2] = np.clip(
                self.player_pos[2], self.range_z[0], self.range_z[1]
            )

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

        # print("No locomotion command issued this frame.")


def main(args=None):
    rclpy.init(args=args)

    node = IntegratedLocomotion()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    # try:
    r = node.create_rate(100.0)  # 100Hz
    while rclpy.ok():
        node.run()
        r.sleep()

    # except KeyboardInterrupt:
    #     node.get_logger().info("Keyboard Interrupt (SIGINT)")
    # except Exception as e:
    #     node.get_logger().error(f"Exception in main loop: {e}")

    # finally:
    th.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
