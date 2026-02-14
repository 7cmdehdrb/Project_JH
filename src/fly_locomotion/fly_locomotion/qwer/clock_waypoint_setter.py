import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *
from std_msgs.msg import *
from enum import Enum
from typing import List, Tuple, Optional, Any
import time
import numpy as np
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


class ClockWaypointSetter(Node):
    def __init__(self):
        super().__init__("clock_waypoint_setter")

        self.__waypoint_pos_reset_pub = self.create_publisher(
            PoseArray,
            "/ball_setter",
            qos_profile=qos_profile_system_default,
        )
        self.__ball_cnt_pub = self.create_publisher(
            Int32,
            "/ball_count",
            qos_profile=qos_profile_system_default,
        )

        self.__player_pos_sub = self.create_subscription(
            PoseStamped,
            "/player_pose",
            self.__player_position_callback,
            qos_profile=qos_profile_system_default,
        )

        self.range = 2.0
        self.ball_count = 12
        self.offset = -1.5

        self.init_player_pos: np.ndarray = None
        self.init_player_rot: np.ndarray = None
        self.player_pose: PoseStamped = None
        self.ball_poses: np.ndarray = None
        self.ball_active: np.ndarray = None  # 각 ball의 활성화 상태

        self.cnt: int = 0
        self.__timer = self.create_timer(0.1, self.main_loop)  # 100ms 주기로 체크

    def __player_position_callback(self, msg: PoseStamped):
        self.player_pose = msg

        if self.init_player_pos is None or self.init_player_rot is None:
            self.init_player_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            )
            self.init_player_rot = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            )
            # 초기화 시 ball 생성
            self.initialize_balls()

    def initialize_balls(self):
        """초기 ball 위치 생성"""
        ball_poses = []

        # player_rot에서 forward 방향 벡터 추출
        rot_matrix = R.from_quat(self.init_player_rot).as_matrix()
        forward_vector = rot_matrix[:, 0]  # x축이 forward 방향

        # forward_vector를 기준으로 좌표계 구성
        world_up = np.array([0.0, 1.0, 0.0])  # Unity의 up 벡터
        right_vector = np.cross(world_up, forward_vector)
        if np.linalg.norm(right_vector) < 0.001:
            right_vector = np.array([0.0, 0.0, 1.0])
        right_vector = right_vector / np.linalg.norm(right_vector)
        up_vector = np.cross(forward_vector, right_vector)
        up_vector = up_vector / np.linalg.norm(up_vector)

        # forward 방향에서 45도 위로 회전한 중심 방향
        elevation_angle = np.radians(90 - 45)
        center_direction = forward_vector * np.cos(
            elevation_angle
        ) + up_vector * np.sin(elevation_angle)
        center_direction = center_direction / np.linalg.norm(center_direction)

        distance = 4.0
        radius = distance * np.cos(elevation_angle)  # 원의 반지름
        center_distance = distance * np.sin(elevation_angle)  # 중심까지의 거리

        for i in range(8):
            # 원형으로 azimuth angle 계산 (0 ~ 360도를 8등분)
            azimuth_angle = (2 * np.pi / 8) * i

            # 중심 방향 주위로 원을 그림
            # right와 up 벡터를 사용하여 원 위의 점 계산
            circle_offset = right_vector * radius * np.cos(
                azimuth_angle
            ) + up_vector * radius * np.sin(azimuth_angle)

            # 최종 위치 = player 위치 + forward 방향으로 이동 + 원 위의 오프셋
            position_world = (
                self.init_player_pos + forward_vector * center_distance + circle_offset
            )

            ball_poses.append(position_world)

        self.ball_poses = np.array(ball_poses)
        self.ball_active = np.ones(8, dtype=bool)  # 모든 ball 활성화

    def main_loop(self):
        """메인 로직: player와 ball의 거리를 체크하고 가까운 ball 제거"""
        if self.player_pose is None or self.ball_poses is None:
            return

        # 현재 player 위치 가져오기
        player_pos = np.array(
            [
                self.player_pose.pose.position.x,
                self.player_pose.pose.position.y,
                self.player_pose.pose.position.z,
            ]
        )

        # 활성화된 ball들과의 거리 계산
        for i in range(len(self.ball_active)):
            if not self.ball_active[i]:
                continue

            # ball과의 거리 계산
            ball_pos = self.ball_poses[i]
            distance = np.linalg.norm(player_pos - ball_pos)
            # print(f"Distance to Ball {i}: {distance:.3f}m")

            # 0.3m 이내에 들어오면 ball 비활성화
            if distance <= 1.0:
                self.ball_active[i] = False
                self.get_logger().info(f"Ball {i} collected! Distance: {distance:.3f}m")
                self.cnt += 1

        # 업데이트된 상태로 poses 발행
        self.publish_ball_poses()
        self.__ball_cnt_pub.publish(Int32(data=self.cnt))

    def publish_ball_poses(self):
        """현재 ball 상태를 발행"""
        if self.ball_poses is None:
            return

        current_time = self.get_clock().now().to_msg()
        poses = []

        for i in range(8):
            if self.ball_active[i]:
                # 활성화된 ball은 원래 위치에
                pose = Pose(
                    position=Point(
                        x=self.ball_poses[i][0],
                        y=self.ball_poses[i][1],
                        z=self.ball_poses[i][2],
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            else:
                # 비활성화된 ball은 z=-999.9로
                pose = Pose(
                    position=Point(x=0.0, y=0.0, z=-999.9),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            poses.append(pose)

        # 나머지 ball도 채우기
        for i in range(8, self.ball_count):
            pose = Pose(
                position=Point(x=0.0, y=0.0, z=-999.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            poses.append(pose)

        msg = PoseArray(
            header=Header(
                stamp=current_time,
                frame_id="world",
            ),
            poses=poses,
        )

        self.__waypoint_pos_reset_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClockWaypointSetter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
