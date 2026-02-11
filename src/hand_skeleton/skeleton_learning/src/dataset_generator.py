#!/usr/bin/env python3

import sys
import rospy
import rosbag
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf.transformations import quaternion_matrix, quaternion_from_matrix
import tkinter as tk
from tkinter import messagebox

matplotlib.use('TkAgg')

def pose2matrix(pose):
    rotation = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    translation = [pose.position.x, pose.position.y, pose.position.z]
    T_matrix = [[rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
                [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
                [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
                [0, 0, 0, 1]]
    return T_matrix

class DatasetGenerator:
    def __init__(self,path,file_name):
        self.bag = rosbag.Bag(path + '/bag_file/' + file_name + '.bag')
        self.dataset = open(path + '/txt_file/' + file_name + '.txt', 'w')
        
        self.msgs = []
        self.index = 0
        self.labels = {}

        try:
            self.load_cache(path + '/cache_file/' + file_name + '.txt')
        except:
            print("No cache file found")
        self.cache = open(path + '/cache_file/' + file_name + '.txt', 'w')

        self.window = tk.Tk()
        self.window.title("Pose Labeling")
        
        # 버튼을 누르면 호출되는 함수
        self.translation_button = tk.Button(self.window, text="Translation", command=lambda: self.process_label('translation'))
        self.translation_button.pack(padx=15, pady=10)
        
        self.rotation_button = tk.Button(self.window, text="Rotation", command=lambda: self.process_label('rotation'))
        self.rotation_button.pack(padx=15, pady=10)

        self.unknown_button = tk.Button(self.window, text="Unknown", command=lambda: self.process_label('unknown'))
        self.unknown_button.pack(padx=15, pady=10)

        self.previous_button = tk.Button(self.window, text="<-", command=lambda: self.process_index(-1))
        self.previous_button.pack(padx=5, pady=10)

        self.next_button = tk.Button(self.window, text="->", command=lambda: self.process_index(1))
        self.next_button.pack(padx=5, pady=10)    
        
    def drawer(self,pose_array, label):
        self.msg = pose_array
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pose in pose_array.poses:
            ax.scatter(pose.position.x, pose.position.y, pose.position.z, c='r', marker='o')

        connections = [
            (0,1),
            (0,2),
            (0,6),
            (0,9),
            (0,12),
            (0,15),

            (2,3),
            (3,4),
            (4,5),
            (5,19),

            (6,7),
            (7,8),
            (8,20),

            (9,10),
            (10,11),
            (11,21),

            (12,13),
            (13,14),
            (14,22),

            (15,16),
            (16,17),
            (17,18),
            (18,23)
        ]

        # 연결 선 그리기
        for start, end in connections:
            start_pose = pose_array.poses[start]
            end_pose = pose_array.poses[end]
            ax.plot([start_pose.position.x, end_pose.position.x],
                    [start_pose.position.y, end_pose.position.y],
                    [start_pose.position.z, end_pose.position.z], c='b')

        plt.title('Index : ' + str(self.index) + '/' + str(len(self.msgs)) + ' Label : ' + label)
        plt.show(block=False)
        self.window.mainloop()
        plt.close(fig)

    def process_label(self, label):
        self.labels[self.index] = label
        self.index += 1
        # UI 창을 닫기 위한 예시 코드
        self.window.quit()

    def process_index(self, index):
        self.index += index
        self.window.quit()

    def frame_change(self,pose_array):
        relative_pose_array = PoseArray()
        relative_pose_array.header.frame_id = "wrist"
        reference = pose_array.poses[0]
        reference_matrix = pose2matrix(reference)
        for pose in pose_array.poses:
            pose_matrix = pose2matrix(pose)
            relative_matrix = np.dot(np.linalg.inv(reference_matrix), pose_matrix)
            relative_pose = Pose()
            relative_pose.position.x = relative_matrix[0][3]
            relative_pose.position.y = relative_matrix[1][3]
            relative_pose.position.z = relative_matrix[2][3]
            quaternion = quaternion_from_matrix(relative_matrix)
            relative_pose.orientation.x = quaternion[0]
            relative_pose.orientation.y = quaternion[1]
            relative_pose.orientation.z = quaternion[2]
            relative_pose.orientation.w = quaternion[3]
            relative_pose_array.poses.append(relative_pose)
        return relative_pose_array
    
    def generate(self):
        print("Generating dataset...")
        self.msgs = []
        for topic, msg, t in self.bag.read_messages(topics=['/l_hand_skeleton_pose']):
            self.msgs.append(msg)
        max_index = len(self.msgs)-1
        while rospy.is_shutdown() == False:
            if self.index >= max_index:
                self.index = max_index
                print("This is the last frame")
            elif self.index < 0:
                self.index = 0
                print("This is the first frame")

            try:
                label = self.labels[self.index]
            except:
                label = "Not labeled"

            print(f"Frame index: {self.index} Frame label: {label}")
            self.drawer(self.frame_change(self.msgs[self.index]), label)
        self.save_dataset(cache=True)
        self.dataset.close()
        self.bag.close()

    def save_dataset(self, cache=False):
        for index, label in self.labels.items():
            self.dataset.write(str(label)+' ')
            poses = self.frame_change(self.msgs[index])
            for pose in poses.poses:
                self.dataset.write(str(pose.position.x)+' '+str(pose.position.y)+' '+str(pose.position.z)+' '+str(pose.orientation.x)+' '+str(pose.orientation.y)+' '+str(pose.orientation.z)+' '+str(pose.orientation.w)+' ')
            self.dataset.write('\n')
        if cache:
            for index, label in self.labels.items():
                self.cache.write(str(index)+' '+str(label)+'\n')
            self.cache.write('lastIndex'+' '+str(self.index)+'\n')

    def load_cache(self, path):
        with open(path, 'r') as cache_file:
            for line in cache_file:
                index, label = line.split()
                if not index == 'lastIndex':
                    self.labels[int(index)] = label
                else:
                    self.index = int(label)

        cache_file.close()

def main():
    rospy.init_node('dataset_generator', anonymous=True)
    dg = DatasetGenerator("/home/hoon/skeleton_dataset","dataset_20241104_1")
    dg.generate()

if __name__ == '__main__':
    main()