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
from visualization_msgs.msg import *

# TF
from tf2_ros import *

# Python
import numpy as np
from rotutils import *


class ManipulatorSpawner(Node):
    def __init__(self):
        super().__init__("manipulator_spawner")

        self.__ans_pub = self.create_publisher(
            PoseArray, "mm_poses", qos_profile=qos_profile_system_default
        )

        self.__err_pub = self.create_publisher(
            PoseArray, "mm_error_poses", qos_profile=qos_profile_system_default
        )

        self.__ans_msg = PoseArray()
        self.__err_msg = PoseArray()

        self.__rng = np.random.default_rng()
        self.__spawn_range = {
            "x": [0.0, 27.5],
            "y": 1.0,
            "z": [[-4.7, -3.7], [-1.7, -0.7], [1.3, 2.3], [4.3, 5.3]],
            "pitch": [-np.pi, np.pi],
        }

    @property
    def message(self) -> Tuple[PoseArray, PoseArray]:
        """
        Returns:
            Tuple[PoseArray, PoseArray]: (correct poses, error poses)
        """
        return self.__ans_msg, self.__err_msg

    def publish(self):
        print("Publishing manipulator poses...")
        print(self.__ans_msg)

        corr_pa = PoseArray(
            header=Header(frame_id="map", stamp=self.get_clock().now().to_msg()),
            poses=[],
        )
        err_pa = PoseArray(
            header=Header(frame_id="map", stamp=self.get_clock().now().to_msg()),
            poses=[],
        )

        for j in range(20):
            dummy_pose = Pose(
                position=Point(x=0.0, y=-999.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            err_pa.poses.append(dummy_pose)
        for j in range(20):
            dummy_pose = Pose(
                position=Point(x=0.0, y=-999.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
            corr_pa.poses.append(dummy_pose)

        self.__ans_msg = corr_pa
        self.__err_msg = err_pa

        self.__ans_pub.publish(self.__ans_msg)
        self.__err_pub.publish(self.__err_msg)

    @staticmethod
    def _l1_xyz(a: Pose, b: Pose) -> float:
        # L1 norm in position space only (x,y,z)
        np_a: np.ndarray = np.array([a.position.x, a.position.y, a.position.z])
        np_b: np.ndarray = np.array([b.position.x, b.position.y, b.position.z])

        return float(np.abs(np_a - np_b).sum())

    def _sample_pose(self) -> Pose:
        x0, x1 = self.__spawn_range["x"]
        y = float(self.__spawn_range["y"])
        z_ranges = self.__spawn_range["z"]
        p0, p1 = self.__spawn_range["pitch"]

        x = self.__rng.uniform(x0, x1)
        zr = z_ranges[self.__rng.integers(0, len(z_ranges))]
        z = self.__rng.uniform(float(zr[0]), float(zr[1]))
        pitch = self.__rng.uniform(p0, p1)

        # Pose = [x, y, z, pitch]
        quat = quaternion_from_euler(0.0, pitch, 0.0)
        p = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(**dict(zip(["x", "y", "z", "w"], quat))),
        )

        return p

    def spawn(self, corr: int, err: int) -> np.ndarray:
        assert corr >= 0 and err >= 0, "corr and err must be non-negative"

        n = corr + err
        if n <= 0:
            return np.empty((0, 4), dtype=np.float64)

        # 전체 세트 생성 실패 시, 전체 Retry
        max_inner_tries = 1000  # 한 세트 안에서 샘플링 시도 한도
        while True:
            poses = []
            corr_pa = PoseArray(
                header=Header(frame_id="map", stamp=self.get_clock().now().to_msg()),
                poses=[],
            )
            err_pa = PoseArray(
                header=Header(frame_id="map", stamp=self.get_clock().now().to_msg()),
                poses=[],
            )

            for j in range(20 - corr):
                dummy_pose = Pose(
                    position=Point(x=0.0, y=-999.0, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
                err_pa.poses.append(dummy_pose)
            for j in range(20 - err):
                dummy_pose = Pose(
                    position=Point(x=0.0, y=-999.0, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
                corr_pa.poses.append(dummy_pose)

            self.__ans_msg = corr_pa
            self.__err_msg = err_pa

            return

            tries = 0

            while len(poses) < n:
                if tries >= max_inner_tries:
                    break  # 공간 부족/운 나쁨 -> 전체 세트 재시도

                candidate: Pose = self._sample_pose()
                ok = True
                for p in poses:
                    p: Pose
                    if self._l1_xyz(candidate, p) < 3.0:
                        ok = False
                        break

                if ok:
                    poses.append(candidate)

                tries += 1

            if len(poses) == n:
                print(f"Generated {n} poses in {tries} tries.")
                for i in range(corr):
                    corr_pa.poses.append(poses[i])
                for i in range(corr, n):
                    err_pa.poses.append(poses[i])
                for j in range(20 - corr):
                    dummy_pose = Pose(
                        position=Point(x=0.0, y=-999.0, z=0.0),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                    err_pa.poses.append(dummy_pose)
                for j in range(20 - err):
                    dummy_pose = Pose(
                        position=Point(x=0.0, y=-999.0, z=0.0),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                    corr_pa.poses.append(dummy_pose)

                print(
                    f"Spawned {corr} correct and {err} error manipulators successfully."
                )

                self.__ans_msg = corr_pa
                self.__err_msg = err_pa

                return None


def main():
    rclpy.init(args=None)

    node = ManipulatorSpawner()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    total = 20

    random_int_2 = np.random.randint(1, 3)
    random_int_2 = np.random.randint(4, 6)
    random_int_2 = np.random.randint(7, 9)
    random_int_1 = total - random_int_2

    random_int_1 = 0
    random_int_2 = 0

    print(
        f"Spawning {random_int_1} correct manipulators and {random_int_2} error manipulators."
    )

    node.spawn(corr=random_int_1, err=random_int_2)

    try:
        r = node.create_rate(0.5)
        while rclpy.ok():
            node.publish()
            r.sleep()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception in main: {e}")
    finally:
        th.join()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
