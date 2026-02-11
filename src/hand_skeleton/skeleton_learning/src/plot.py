#!/usr/bin/env python3

import rospy
import numpy as np
import rosbag
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from tf.transformations import *
import matplotlib.pyplot as plt

def quaternion_to_euler(quaternion, order='zyx', input_order='xyzw'):
    """
    Convert a quaternion to Euler angles.

    Parameters:
    - quaternion: A list or array [w, x, y, z] representing the quaternion.
    - order: The order of the Euler angles, default is 'zyx' (yaw, pitch, roll).

    Returns:
    - euler_angles: A tuple (roll, pitch, yaw) in radians.
    """
    if input_order == 'xyzw':
        x, y, z, w = quaternion
        quaternion = np.array([w, x, y, z])
    elif input_order == 'wxyz':
        pass  # Already in [w, x, y, z] order
    else:
        raise ValueError("Unsupported input_order. Use 'xyzw' or 'wxyz'.")

    # Normalize the quaternion
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Invalid quaternion: norm is zero.")
    quaternion /= norm  # Normalize the quaternion

    w, x, y, z = quaternion
    
    # Compute yaw (Z-axis rotation)
    if order == 'zyx':
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    elif order == 'xyz':  # For roll-pitch-yaw order
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    else:
        raise ValueError("Unsupported Euler angle order. Use 'zyx' or 'xyz'.")
    
    return roll, pitch, yaw

class Plotter:
    def __init__(self):
        self.bag = rosbag.Bag('/home/hoon/locomotion_test/2024-12-03-10-50-56.bag')
        self.f = open('/home/hoon/locomotion_test/2024-12-03-10-50-56.txt', 'w')
        self.poses = [[],[],[],[],[],[],[]]
        self.posesTime = []
        self.rpy = [[],[],[]]
        self.rpyTime = []
        self.labels = [[],[],[]]
        self.labelsTime = []
        self.skeleton = [[],[],[],[],[],[],[]]
        self.skeletonTime = []
        self.relativeSkeleton = [[],[],[],[],[],[],[]]
        self.relativeSkeletonTime = []
        self.skeletonRPY = [[],[],[]]
        self.skeletonRPYTime = []
        self.relativeSkeletonRPY = [[],[],[]]
        self.relativeSkeletonRPYTime = []
        self.deltaPose = [[],[],[],[],[],[],[]]
        self.deltaPoseTime = []
        self.deltaPoseRPY = [[],[],[]]
        self.deltaPoseRPYTime = []
        self.posesFist = [[],[],[],[],[],[],[]]
        self.posesTimeFist = []
        self.posesPointing = [[],[],[],[],[],[],[]]
        self.posesTimePointing = []
        self.posesUnknown = [[],[],[],[],[],[],[]]
        self.posesTimeUnknown = []
        self.loadDataFromBag()

        

    def loadDataFromBag(self):
        for topic, msg, t in self.bag.read_messages(topics=['/player_pose','/gesture_array','/l_hand_skeleton_pose','/l_wrist_pose','/delta_pose']):
            if topic == '/player_pose':
                self.poses[0].append(msg.pose.position.x)
                self.poses[1].append(msg.pose.position.y)
                self.poses[2].append(msg.pose.position.z)
                self.poses[3].append(msg.pose.orientation.x)
                self.poses[4].append(msg.pose.orientation.y)
                self.poses[5].append(msg.pose.orientation.z)
                self.poses[6].append(msg.pose.orientation.w)
                self.posesTime.append(t.to_sec())
            elif topic == '/gesture_array':
                self.labels[0].append(msg.data[0])
                self.labels[1].append(msg.data[1])
                self.labels[2].append(msg.data[2])
                self.labelsTime.append(t.to_sec())
            elif topic == '/l_hand_skeleton_pose':
                self.skeleton[0].append(msg.poses[0].position.x)
                self.skeleton[1].append(msg.poses[0].position.y)
                self.skeleton[2].append(msg.poses[0].position.z)
                self.skeleton[3].append(msg.poses[0].orientation.x)
                self.skeleton[4].append(msg.poses[0].orientation.y)
                self.skeleton[5].append(msg.poses[0].orientation.z)
                self.skeleton[6].append(msg.poses[0].orientation.w)
                self.skeletonTime.append(t.to_sec())
            elif topic == '/l_wrist_pose':
                self.relativeSkeleton[0].append(msg.pose.position.x)
                self.relativeSkeleton[1].append(msg.pose.position.y)
                self.relativeSkeleton[2].append(msg.pose.position.z)
                self.relativeSkeleton[3].append(msg.pose.orientation.x)
                self.relativeSkeleton[4].append(msg.pose.orientation.y)
                self.relativeSkeleton[5].append(msg.pose.orientation.z)
                self.relativeSkeleton[6].append(msg.pose.orientation.w)
                self.relativeSkeletonTime.append(t.to_sec())
            elif topic == '/delta_pose':
                self.deltaPose[0].append(msg.pose.position.x)
                self.deltaPose[1].append(msg.pose.position.y)
                self.deltaPose[2].append(msg.pose.position.z)
                self.deltaPose[3].append(msg.pose.orientation.x)
                self.deltaPose[4].append(msg.pose.orientation.y)
                self.deltaPose[5].append(msg.pose.orientation.z)
                self.deltaPose[6].append(msg.pose.orientation.w)
                self.deltaPoseTime.append(t.to_sec())

        # for pose in self.poses:
        #     for p in pose:
        #         self.f.write(str(p) + ' ')
        #     self.f.write('\n')
        # for time in self.posesTime:
        #     self.f.write(str(time) + ' ')
        # self.f.write('\n')

        # for label in self.labels:
        #     for l in label:
        #         self.f.write(str(l) + ' ')
        #     self.f.write('\n')
        # for time in self.labelsTime:
        #     self.f.write(str(time) + ' ')
        # self.f.write('\n')

        # for skel in self.skeleton:
        #     for s in skel:
        #         self.f.write(str(s) + ' ')
        #     self.f.write('\n')
        # for time in self.skeletonTime:
        #     self.f.write(str(time) + ' ')
        # self.f.write('\n')

        # self.f.close()

        for i in range(len(self.poses[0])):
            qx = self.poses[3][i]
            qy = self.poses[4][i]
            qz = self.poses[5][i]
            qw = self.poses[6][i]
            roll, pitch, yaw = quaternion_to_euler([qx, qy, qz, qw])
            self.rpy[0].append(roll)
            self.rpy[1].append(pitch)
            self.rpy[2].append(yaw)
            self.rpyTime.append(self.posesTime[i])

        for i in range(len(self.skeleton[0])):
            qx = self.skeleton[3][i]
            qy = self.skeleton[4][i]
            qz = self.skeleton[5][i]
            qw = self.skeleton[6][i]
            roll, pitch, yaw = quaternion_to_euler([qx, qy, qz, qw])
            self.skeletonRPY[0].append(roll)
            self.skeletonRPY[1].append(pitch)
            self.skeletonRPY[2].append(yaw)
            self.skeletonRPYTime.append(self.skeletonTime[i])

        for i in range(len(self.relativeSkeleton[0])):
            qx = self.relativeSkeleton[3][i]
            qy = self.relativeSkeleton[4][i]
            qz = self.relativeSkeleton[5][i]
            qw = self.relativeSkeleton[6][i]
            roll, pitch, yaw = quaternion_to_euler([qx, qy, qz, qw])
            self.relativeSkeletonRPY[0].append(roll)
            self.relativeSkeletonRPY[1].append(pitch)
            self.relativeSkeletonRPY[2].append(yaw)
            self.relativeSkeletonRPYTime.append(self.relativeSkeletonTime[i])
        
        for i in range(len(self.deltaPose[0])):
            qx = self.deltaPose[3][i]
            qy = self.deltaPose[4][i]
            qz = self.deltaPose[5][i]
            qw = self.deltaPose[6][i]
            roll, pitch, yaw = quaternion_to_euler([qx, qy, qz, qw])
            self.deltaPoseRPY[0].append(roll)
            self.deltaPoseRPY[1].append(pitch)
            self.deltaPoseRPY[2].append(yaw)
            self.deltaPoseRPYTime.append(self.deltaPoseTime[i])

        for i, pose_time in enumerate(self.posesTime):
            closest_label_idx = np.argmin(np.abs(np.array(self.labelsTime) - pose_time))
            
            label_idx = np.argmax([
                self.labels[0][closest_label_idx], 
                self.labels[1][closest_label_idx], 
                self.labels[2][closest_label_idx]
            ])
            if label_idx == 0:
                self.posesFist[0].append(self.poses[0][i])
                self.posesFist[1].append(self.poses[1][i])
                self.posesFist[2].append(self.poses[2][i])
                self.posesFist[3].append(self.poses[3][i])
                self.posesFist[4].append(self.poses[4][i])
                self.posesFist[5].append(self.poses[5][i])
                self.posesFist[6].append(self.poses[6][i])
                self.posesTimeFist.append(pose_time)
            elif label_idx == 1:
                self.posesPointing[0].append(self.poses[0][i])
                self.posesPointing[1].append(self.poses[1][i])
                self.posesPointing[2].append(self.poses[2][i])
                self.posesPointing[3].append(self.poses[3][i])
                self.posesPointing[4].append(self.poses[4][i])
                self.posesPointing[5].append(self.poses[5][i])
                self.posesPointing[6].append(self.poses[6][i])
                self.posesTimePointing.append(pose_time)
            else:
                self.posesUnknown[0].append(self.poses[0][i])
                self.posesUnknown[1].append(self.poses[1][i])
                self.posesUnknown[2].append(self.poses[2][i])
                self.posesUnknown[3].append(self.poses[3][i])
                self.posesUnknown[4].append(self.poses[4][i])
                self.posesUnknown[5].append(self.poses[5][i])
                self.posesUnknown[6].append(self.poses[6][i])
                self.posesTimeUnknown.append(pose_time)
        print(len(self.posesFist[0]), len(self.posesPointing[0]), len(self.posesUnknown[0]))


    def plotTranslation(self):
        print(len(self.posesTime), len(self.skeletonTime), len(self.labelsTime), len(self.relativeSkeletonTime), len(self.deltaPoseTime))
        # fig = plt.figure()
        # ax = fig.add_subplot(1,2,1,projection='3d')
        # ax.plot3D(self.poses[0],self.poses[1],self.poses[2], label='Player Pose')
        # # ax.plot3D(self.poses[0],self.poses[1],np.zeros(len(self.poses[2])), label='Player Pose')
        # # ax.plot3D(self.poses[0],np.zeros(len(self.poses[1])),self.poses[2], label='Player Pose')
        # # ax.plot3D(np.zeros(len(self.poses[0])),self.poses[1],self.poses[2], label='Player Pose')
        # # ax.scatter3D(self.posesFist[0],self.posesFist[1],self.posesFist[2], label='Fist Pose', s=1)
        # # ax.scatter3D(self.posesPointing[0],self.posesPointing[1],self.posesPointing[2], label='Pointing Pose', s=1)
        # # ax.scatter3D(self.posesUnknown[0],self.posesUnknown[1],self.posesUnknown[2], label='Unknown Pose', s=1)
        # ax.plot3D(self.skeleton[0],self.skeleton[1],self.skeleton[2], label='Wrist Pose')
        # ax.axis('equal')
        # ax.legend()

        # ax2 = fig.add_subplot(3,2,2)
        # ax2.plot(self.posesTime,self.poses[0],label='x')
        # ax2.plot(self.relativeSkeletonTime,self.relativeSkeleton[0],label='skeleton x')
        # ax2.plot(self.labelsTime,self.labels[0],label='Fist')
        # ax2.plot(self.labelsTime,self.labels[1],label='Pointing')
        # # ax2.plot(self.labelsTime,self.labels[2],label='Unknown')
        # ax2.legend()
        # ax2.grid(True)
        # # ax2.axis('equal')

        # ax3 = fig.add_subplot(3,2,4)
        # ax3.plot(self.posesTime,self.poses[1],label='y')
        # ax3.plot(self.relativeSkeletonTime,self.relativeSkeleton[1],label='skeleton y')
        # ax3.plot(self.labelsTime,self.labels[0],label='Fist')
        # ax3.plot(self.labelsTime,self.labels[1],label='Pointing')
        # # ax3.plot(self.labelsTime,self.labels[2],label='Unknown')
        # ax3.legend()
        # ax3.grid(True)
        # # ax3.axis('equal')

        # ax4 = fig.add_subplot(3,2,6)
        # ax4.plot(self.posesTime,self.poses[2],label='z')
        # ax4.plot(self.relativeSkeletonTime,self.relativeSkeleton[2],label='skeleton z')
        # ax4.plot(self.labelsTime,self.labels[0],label='Fist')
        # ax4.plot(self.labelsTime,self.labels[1],label='Pointing')
        # # ax4.plot(self.labelsTime,self.labels[2],label='Unknown')
        # ax4.legend()
        # ax4.grid(True)
        # # ax4.axis('equal')

        # plt.show()

        # fig2 = plt.figure()
        # ax5 = fig2.add_subplot(1,2,1,projection='3d')
        # ax5.plot3D(self.poses[0],self.poses[1],self.poses[2], label='Player Pose')
        # # ax5.scatter3D(self.posesFist[0],self.posesFist[1],self.posesFist[2], label='Fist Pose', s=1)
        # # ax5.scatter3D(self.posesPointing[0],self.posesPointing[1],self.posesPointing[2], label='Pointing Pose', s=1)
        # # ax5.scatter3D(self.posesUnknown[0],self.posesUnknown[1],self.posesUnknown[2], label='Unknown Pose', s=1)
        # ax5.plot3D(self.relativeSkeleton[0],self.relativeSkeleton[1],self.relativeSkeleton[2], label='Wrist Pose')
        # ax5.axis('equal')
        # ax5.legend()

        # ax6 = fig2.add_subplot(3,2,2)
        # ax6.plot(self.rpyTime,self.rpy[0],label='roll')
        # ax6.plot(self.relativeSkeletonRPYTime,self.relativeSkeletonRPY[0],label='skeleton roll')
        # ax6.plot(self.labelsTime,self.labels[1],label='Pointing')
        # ax6.legend()
        # ax6.grid(True)

        # ax7 = fig2.add_subplot(3,2,4)
        # ax7.plot(self.rpyTime,self.rpy[1],label='pitch')
        # ax7.plot(self.relativeSkeletonRPYTime,self.relativeSkeletonRPY[1],label='skeleton pitch')
        # ax7.plot(self.labelsTime,self.labels[1],label='Pointing')
        # ax7.legend()
        # ax7.grid(True)

        # ax8 = fig2.add_subplot(3,2,6)
        # ax8.plot(self.rpyTime,self.rpy[2],label='yaw')
        # ax8.plot(self.relativeSkeletonRPYTime,self.relativeSkeletonRPY[2],label='skeleton yaw')
        # ax8.plot(self.labelsTime,self.labels[1],label='Pointing')
        # ax8.legend()
        # ax8.grid(True)

        # plt.show()

        fig3 = plt.figure()
        # ax10 = fig3.add_subplot(1,2,1,projection='3d')
        # ax10.plot3D(self.poses[0],self.poses[1],self.poses[2], label='Player Pose')
        # ax10.plot3D(self.deltaPose[0],self.deltaPose[1],self.deltaPose[2], label='Delta Pose')
        # ax10.axis('equal')
        # ax10.legend()

        ax11 = fig3.add_subplot(3,2,1)
        ax11.plot(self.posesTime,self.poses[0],label='x')
        ax11.scatter(self.deltaPoseTime,self.deltaPose[0],label='delta x',s=0.5,c='r')
        ax11.plot(self.labelsTime,self.labels[0],label='Fist')
        ax11.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax11.legend()
        ax11.set_ylim(-3.5,3.5)
        ax11.grid(True)

        ax12 = fig3.add_subplot(3,2,3)
        ax12.plot(self.posesTime,self.poses[1],label='y')
        ax12.scatter(self.deltaPoseTime,self.deltaPose[1],label='delta y',s=0.5,c='r')
        ax12.plot(self.labelsTime,self.labels[0],label='Fist')
        ax12.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax12.legend()
        ax12.set_ylim(-3.5,3.5)
        ax12.grid(True)

        ax13 = fig3.add_subplot(3,2,5)
        ax13.plot(self.posesTime,self.poses[2],label='z')
        ax13.scatter(self.deltaPoseTime,self.deltaPose[2],label='delta z',s=0.5,c='r')
        ax13.plot(self.labelsTime,self.labels[0],label='Fist')
        ax13.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax13.legend()
        ax13.set_ylim(-3.5,3.5)
        ax13.grid(True)


        # fig4 = plt.figure()
        # ax14 = fig4.add_subplot(1,2,1,projection='3d')
        # ax14.plot3D(self.poses[0],self.poses[1],self.poses[2], label='Player Pose')
        # ax14.plot3D(self.deltaPose[0],self.deltaPose[1],self.deltaPose[2], label='Wrist Pose')
        # ax14.axis('equal')
        # ax14.legend()

        ax15 = fig3.add_subplot(3,2,2)
        ax15.plot(self.rpyTime,self.rpy[0],label='roll')
        ax15.scatter(self.deltaPoseRPYTime,self.deltaPoseRPY[0],label='delta roll',s=0.5, c='r')
        ax15.plot(self.labelsTime,self.labels[0],label='Fist')
        ax15.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax15.legend()
        ax15.set_ylim(-3.5,3.5)
        ax15.grid(True)

        ax16 = fig3.add_subplot(3,2,4)
        ax16.plot(self.rpyTime,self.rpy[1],label='pitch')
        ax16.scatter(self.deltaPoseRPYTime,self.deltaPoseRPY[1],label='delta pitch',s=0.5, c='r')
        ax16.plot(self.labelsTime,self.labels[0],label='Fist')
        ax16.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax16.legend()
        ax16.set_ylim(-3.5,3.5)
        ax16.grid(True)

        ax17 = fig3.add_subplot(3,2,6)
        ax17.plot(self.rpyTime,self.rpy[2],label='yaw')
        ax17.scatter(self.deltaPoseRPYTime,self.deltaPoseRPY[2],label='delta yaw',s=0.5, c='r')
        ax17.plot(self.labelsTime,self.labels[0],label='Fist')
        ax17.plot(self.labelsTime,self.labels[1],label='Pointing')
        ax17.legend()
        ax17.set_ylim(-3.5,3.5)
        ax17.grid(True)

        plt.show()


def main():
    rospy.init_node('plotter', anonymous=True)
    plotter = Plotter()
    plotter.plotTranslation()

if __name__ == '__main__':
    main()