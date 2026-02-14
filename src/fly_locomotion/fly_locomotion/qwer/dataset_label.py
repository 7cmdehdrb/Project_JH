#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tkinter as tk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from geometry_msgs.msg import PoseArray, Pose

# ROS2 message type resolving
from rosidl_runtime_py.utilities import get_message


# ----------------------------
# math utilities (no ROS1 tf)
# ----------------------------
def quat_to_rotmat(x, y, z, w):
    # normalized quaternion -> 3x3
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    R = np.array([
        [1.0 - (yy + zz),       (xy - wz),       (xz + wy)],
        [      (xy + wz), 1.0 - (xx + zz),       (yz - wx)],
        [      (xz - wy),       (yz + wx), 1.0 - (xx + yy)],
    ], dtype=np.float64)
    return R


def rotmat_to_quat(R):
    # 3x3 -> (x,y,z,w)
    # stable enough for this use
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=np.float64)
    # normalize
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def pose_to_T(p: Pose):
    R = quat_to_rotmat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    t = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ----------------------------
# Dataset generator (ROS2)
# ----------------------------
class DatasetGeneratorROS2:
    def __init__(self, root_path: str, bag_dir_name: str, topic_name: str = "/l_hand_skeleton_pose"):
        self.root_path = root_path
        self.bag_dir = "/home/hoon/fligt_ws/rosbag2_2026_02_11-15_27_48"  # bag is a DIRECTORY in ROS2
        self.topic = topic_name

        os.makedirs(os.path.join(root_path, "txt_file"), exist_ok=True)
        os.makedirs(os.path.join(root_path, "cache_file"), exist_ok=True)

        self.dataset_path = os.path.join(root_path, "txt_file", f"{bag_dir_name}.txt")
        self.cache_path = os.path.join(root_path, "cache_file", f"{bag_dir_name}.txt")

        self.dataset = open(self.dataset_path, "w", encoding="utf-8")

        self.msgs = []   # list[PoseArray]
        self.index = 0
        self.labels = {}  # dict[int,str]

        # load cache if exists
        if os.path.exists(self.cache_path):
            self.load_cache(self.cache_path)
        self.cache = open(self.cache_path, "w", encoding="utf-8")

        # UI
        self.window = tk.Tk()
        self.window.title("Pose Labeling (ROS2)")

        self.translation_button = tk.Button(self.window, text="Translation", command=lambda: self.process_label("translation"))
        self.translation_button.pack(padx=15, pady=8)

        self.rotation_button = tk.Button(self.window, text="Rotation", command=lambda: self.process_label("rotation"))
        self.rotation_button.pack(padx=15, pady=8)

        self.unknown_button = tk.Button(self.window, text="Unknown", command=lambda: self.process_label("unknown"))
        self.unknown_button.pack(padx=15, pady=8)

        self.twofinger_button = tk.Button(self.window, text="TwoFinger", command=lambda: self.process_label("twofinger"))
        self.twofinger_button.pack(padx=15, pady=8)

        self.previous_button = tk.Button(self.window, text="<-", command=lambda: self.process_index(-1))
        self.previous_button.pack(padx=5, pady=8)

        self.next_button = tk.Button(self.window, text="->", command=lambda: self.process_index(1))
        self.next_button.pack(padx=5, pady=8)

        # optional: 키보드도 지원
        self.window.bind("<Left>", lambda e: self.process_index(-1))
        self.window.bind("<Right>", lambda e: self.process_index(1))
        self.window.bind("t", lambda e: self.process_label("translation"))
        self.window.bind("r", lambda e: self.process_label("rotation"))
        self.window.bind("u", lambda e: self.process_label("unknown"))
        self.window.bind("2", lambda e: self.process_label("twofinger"))

    def read_bag_posearrays(self):
        if not os.path.isdir(self.bag_dir):
            raise FileNotFoundError(f"Bag directory not found: {self.bag_dir}")

        reader = SequentialReader()
        storage_options = StorageOptions(uri=self.bag_dir, storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader.open(storage_options, converter_options)

        topics_and_types = reader.get_all_topics_and_types()
        type_map = {tt.name: tt.type for tt in topics_and_types}

        if self.topic not in type_map:
            raise RuntimeError(f"Topic '{self.topic}' not found in bag. Available topics:\n{list(type_map.keys())}")

        msg_type_str = type_map[self.topic]
        msg_type = get_message(msg_type_str)

        msgs = []
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic != self.topic:
                continue
            msg = deserialize_message(data, msg_type)
            # 안전장치: PoseArray만 받는다고 가정
            if not hasattr(msg, "poses"):
                continue
            msgs.append(msg)

        return msgs

    def frame_change(self, pose_array: PoseArray):
        relative_pose_array = PoseArray()
        relative_pose_array.header.frame_id = "wrist"

        reference = pose_array.poses[0]
        T_ref = pose_to_T(reference)
        T_ref_inv = np.linalg.inv(T_ref)

        for p in pose_array.poses:
            T = pose_to_T(p)
            T_rel = T_ref_inv @ T

            rel_pose = Pose()
            rel_pose.position.x = float(T_rel[0, 3])
            rel_pose.position.y = float(T_rel[1, 3])
            rel_pose.position.z = float(T_rel[2, 3])

            q = rotmat_to_quat(T_rel[:3, :3])
            rel_pose.orientation.x = float(q[0])
            rel_pose.orientation.y = float(q[1])
            rel_pose.orientation.z = float(q[2])
            rel_pose.orientation.w = float(q[3])

            relative_pose_array.poses.append(rel_pose)

        return relative_pose_array

    def drawer(self, pose_array: PoseArray, label: str):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        xs, ys, zs = [], [], []
        for p in pose_array.poses:
            xs.append(p.position.x)
            ys.append(p.position.y)
            zs.append(p.position.z)

        ax.scatter(xs, ys, zs, marker="o")

        # ROS1 코드랑 동일한 연결 정의
        connections = [
            (0, 1), (0, 2), (0, 6), (0, 9), (0, 12), (0, 15),
            (2, 3), (3, 4), (4, 5), (5, 19),
            (6, 7), (7, 8), (8, 20),
            (9, 10), (10, 11), (11, 21),
            (12, 13), (13, 14), (14, 22),
            (15, 16), (16, 17), (17, 18), (18, 23),
        ]

        for s, e in connections:
            if s >= len(pose_array.poses) or e >= len(pose_array.poses):
                continue
            ps = pose_array.poses[s].position
            pe = pose_array.poses[e].position
            ax.plot([ps.x, pe.x], [ps.y, pe.y], [ps.z, pe.z])

        plt.title(f"Index: {self.index}/{len(self.msgs)-1}   Label: {label}")
        ax.tick_params(axis='both', which='major', labelsize=10)
        # ax.xlim([0.0, 0.2])
        # ax.ylim([0.0, 0.2])
        # ax.zlim([0.0, 0.2])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
        plt.show(block=False)

        # Tk 이벤트 루프를 돌려서 버튼 클릭을 기다림
        self.window.mainloop()

        plt.close(fig)

    def process_label(self, label: str):
        self.labels[self.index] = label
        self.index += 1
        self.window.quit()

    def process_index(self, delta: int):
        self.index += delta
        self.window.quit()

    def save_dataset(self, cache: bool = False):
        # dataset txt
        for idx in sorted(self.labels.keys()):
            label = self.labels[idx]
            self.dataset.write(str(label) + " ")

            poses = self.frame_change(self.msgs[idx])
            for p in poses.poses:
                self.dataset.write(
                    f"{p.position.x} {p.position.y} {p.position.z} "
                    f"{p.orientation.x} {p.orientation.y} {p.orientation.z} {p.orientation.w} "
                )
            self.dataset.write("\n")

        # cache txt
        if cache:
            for idx in sorted(self.labels.keys()):
                self.cache.write(f"{idx} {self.labels[idx]}\n")
            self.cache.write(f"lastIndex {self.index}\n")

    def load_cache(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                k, v = parts
                if k == "lastIndex":
                    self.index = int(v)
                else:
                    self.labels[int(k)] = v

    def generate(self):
        print("Reading bag...")
        self.msgs = self.read_bag_posearrays()
        if len(self.msgs) == 0:
            raise RuntimeError(f"No messages read from topic '{self.topic}'.")

        max_index = len(self.msgs) - 1
        print(f"Loaded {len(self.msgs)} frames from {self.topic}")

        # Offline loop
        while rclpy.ok():
            if self.index >= max_index:
                self.index = max_index
                print("This is the last frame")
            elif self.index < 0:
                self.index = 0
                print("This is the first frame")

            label = self.labels.get(self.index, "Not labeled")
            print(f"Frame index: {self.index}  Frame label: {label}")

            pose_rel = self.frame_change(self.msgs[self.index])
            self.drawer(pose_rel, label)

        # 종료 시 저장
        self.save_dataset(cache=True)
        self.dataset.close()
        self.cache.close()


def main():
    rclpy.init(args=None)

    # 너 코드랑 동일한 느낌으로 경로 지정
    root = "/home/hoon/fligt_ws/rosbag2_2026_02_11-15_27_48"
    bag_dir_name = "dataset_20241104_1"  # ROS2 bag "directory" name under root/bag_file/

    dg = DatasetGeneratorROS2(root, bag_dir_name, topic_name="/l_hand_skeleton_pose")
    try:
        dg.generate()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving cache+dataset...")
        dg.save_dataset(cache=True)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
