#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String, Float32MultiArray
from tensorflow.keras.models import load_model
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix

def pose2matrix(pose):
    rotation = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    translation = [pose.position.x, pose.position.y, pose.position.z]
    T_matrix = [[rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
                [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
                [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
                [0, 0, 0, 1]]
    return T_matrix


class SkeletonClassifier:
    def __init__(self):
        self.model = load_model('/home/hoon/skeleton_dataset/model/model_train_241121.h5')
        self.sub = rospy.Subscriber('/l_hand_skeleton_pose', PoseArray, self.callback)
        self.pub = rospy.Publisher('/gesture', String, queue_size=10)
        self.pub_array = rospy.Publisher('/gesture_array', Float32MultiArray, queue_size=10)

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

    def callback(self, msg):
        poses = self.frame_change(msg)
        skeleton_data = []
        for pose in poses.poses:
            skeleton_data.append(pose.position.x)
            skeleton_data.append(pose.position.y)
            skeleton_data.append(pose.position.z)
            skeleton_data.append(pose.orientation.x)
            skeleton_data.append(pose.orientation.y)
            skeleton_data.append(pose.orientation.z)
            skeleton_data.append(pose.orientation.w)
        skeleton_data = np.array([skeleton_data]).astype(float)
        print(skeleton_data.shape)
        prediction = self.model.predict(skeleton_data)
        predicted_label = np.argmax(prediction)
        print(predicted_label, prediction, end=' ')
        label_array = Float32MultiArray()
        label_array.data = prediction[0].tolist()
        print(label_array.data)
        self.pub_array.publish(label_array)
        if predicted_label == 0:
            label = "Translation"
            self.pub.publish(label)
        elif predicted_label == 1:
            label = "Rotation"
            self.pub.publish(label)
        else:
            label = "Unknown"
            self.pub.publish(label)
        

def main():
    rospy.init_node('skeleton_classifier')
    sc = SkeletonClassifier()
    rospy.spin()

if __name__ == '__main__':
    main()
