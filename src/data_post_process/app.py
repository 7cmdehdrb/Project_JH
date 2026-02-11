# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys


# %%
def plot_player_pos(player_df: pd.DataFrame, waypoint_df: pd.DataFrame):
    """
    time,pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w
    time,poses[0].position.x,poses[0].position.y,poses[0].position.z,poses[0].orientation.x,poses[0].orientation.y,poses[0].orientation.z,poses[0].orientation.w,poses[1].position.x,poses[1].position.y,poses[1].position.z,poses[1].orientation.x,poses[1].orientation.y,poses[1].orientation.z,poses[1].orientation.w,poses[2].position.x,poses[2].position.y,poses[2].position.z,poses[2].orientation.x,poses[2].orientation.y,poses[2].orientation.z,poses[2].orientation.w,poses[3].position.x,poses[3].position.y,poses[3].position.z,poses[3].orientation.x,poses[3].orientation.y,poses[3].orientation.z,poses[3].orientation.w,poses[4].position.x,poses[4].position.y,poses[4].position.z,poses[4].orientation.x,poses[4].orientation.y,poses[4].orientation.z,poses[4].orientation.w,poses[5].position.x,poses[5].position.y,poses[5].position.z,poses[5].orientation.x,poses[5].orientation.y,poses[5].orientation.z,poses[5].orientation.w,poses[6].position.x,poses[6].position.y,poses[6].position.z,poses[6].orientation.x,poses[6].orientation.y,poses[6].orientation.z,poses[6].orientation.w,poses[7].position.x,poses[7].position.y,poses[7].position.z,poses[7].orientation.x,poses[7].orientation.y,poses[7].orientation.z,poses[7].orientation.w,poses[8].position.x,poses[8].position.y,poses[8].position.z,poses[8].orientation.x,poses[8].orientation.y,poses[8].orientation.z,poses[8].orientation.w,poses[9].position.x,poses[9].position.y,poses[9].position.z,poses[9].orientation.x,poses[9].orientation.y,poses[9].orientation.z,poses[9].orientation.w,poses[10].position.x,poses[10].position.y,poses[10].position.z,poses[10].orientation.x,poses[10].orientation.y,poses[10].orientation.z,poses[10].orientation.w,poses[11].position.x,poses[11].position.y,poses[11].position.z,poses[11].orientation.x,poses[11].orientation.y,poses[11].orientation.z,poses[11].orientation.w
    """
    times = player_df["time"]
    xs = player_df["pose.position.x"].to_numpy()
    ys = player_df["pose.position.y"].to_numpy()
    zs = player_df["pose.position.z"].to_numpy()

    waypoint_positions = [
        [
            waypoint_df[f"poses[{i}].position.x"].to_numpy()[0],
            waypoint_df[f"poses[{i}].position.y"].to_numpy()[0],
            waypoint_df[f"poses[{i}].position.z"].to_numpy()[0],
        ]
        for i in range(12)
    ]

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, label="Player Position", color="blue", alpha=0.7)

    waypoint_xs = [pos[0] for pos in waypoint_positions]
    waypoint_ys = [pos[1] for pos in waypoint_positions]
    waypoint_zs = [pos[2] for pos in waypoint_positions]
    ax.scatter(
        waypoint_xs, waypoint_ys, waypoint_zs, label="Waypoints", color="red", s=100
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("Player Position and Waypoints")

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    ax.legend()
    plt.show()


def plot_player_pos_2d(player_df: pd.DataFrame, waypoint_df: pd.DataFrame):
    times = player_df["time"]
    xs = player_df["pose.position.x"].to_numpy()
    ys = player_df["pose.position.y"].to_numpy()
    zs = player_df["pose.position.z"].to_numpy()

    waypoint_positions = [
        [
            waypoint_df[f"poses[{i}].position.x"].to_numpy()[0],
            waypoint_df[f"poses[{i}].position.y"].to_numpy()[0],
            waypoint_df[f"poses[{i}].position.z"].to_numpy()[0],
        ]
        for i in range(12)
    ]

    fig = plt.figure(figsize=(10, 10))

    plt.plot(xs, ys, label="Player Position", color="blue", alpha=0.7)

    plt.show()


# %%
bag_folder = "/home/min/7cmdehdrb/fuck_flight/rosbag2_2026_02_10-17_46_40"

csv_files = os.listdir(bag_folder)

player_pos = "player_pose.csv"
waypoints = "waypoint_array.csv"

player_pos_df = pd.read_csv(os.path.join(bag_folder, player_pos))
waypoints_df = pd.read_csv(os.path.join(bag_folder, waypoints))

plot_player_pos(player_pos_df, waypoints_df)
# plot_player_pos_2d(player_pos_df, waypoints_df)
