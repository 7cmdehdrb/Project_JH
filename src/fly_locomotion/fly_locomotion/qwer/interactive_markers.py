# Copyright (c) 2011, Willow Garage, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# Author: Michael Ferguson

from threading import Lock

from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from std_msgs.msg import Header
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerFeedback
from visualization_msgs.msg import InteractiveMarkerPose
from visualization_msgs.msg import InteractiveMarkerUpdate
from visualization_msgs.srv import GetInteractiveMarkers


class MarkerContext:
    """Represents a single marker."""

    def __init__(self, time):
        self.last_feedback = time
        self.last_client_id = ''
        self.default_feedback_callback = None
        self.feedback_callbacks = {}
        self.int_marker = InteractiveMarker()


class UpdateContext:
    """Represents an update to a single marker."""

    FULL_UPDATE = 0
    POSE_UPDATE = 1
    ERASE = 2

    def __init__(self):
        self.update_type = self.FULL_UPDATE
        self.int_marker = InteractiveMarker()
        self.default_feedback_callback = None
        self.feedback_callbacks = {}


class InteractiveMarkerServer:
    """
    A server to one or many clients (e.g. rviz) displaying a set of interactive markers.

    Note: Keep in mind that changes made by calling insert(), erase(), setCallback() etc.
          are not applied until calling applyChanges().
    """

    DEFAULT_FEEDBACK_CALLBACK = 255

    def __init__(
        self,
        node,
        namespace,
        *,
        update_pub_qos=QoSProfile(depth=100),
        feedback_sub_qos=QoSProfile(depth=1)
    ):
        """
        Create an InteractiveMarkerServer and associated ROS connections.

        :param node: The node to attach this interactive marker server to.
        :param namespace: The communication namespace of the interactie marker server.
            Clients that want to interact should connect with the same namespace.
        :param update_pub_qos: QoS settings for the update publisher.
        :param feedback_sub_qos: QoS settings for the feedback subscription.
        """
        self.node = node
        self.namespace = namespace
        self.seq_num = 0
        self.mutex = Lock()

        # contains the current state of all markers
        # string : MarkerContext
        self.marker_contexts = {}

        # updates that have to be sent on the next publish
        # string : UpdateContext
        self.pending_updates = {}

        get_interactive_markers_service_name = namespace + '/get_interactive_markers'
        update_topic = namespace + '/update'
        feedback_topic = namespace + '/feedback'

        self.get_interactive_markers_srv = self.node.create_service(
            GetInteractiveMarkers,
            get_interactive_markers_service_name,
            self.getInteractiveMarkersCallback
        )

        self.update_pub = self.node.create_publisher(
            InteractiveMarkerUpdate,
            update_topic,
            update_pub_qos
        )

        self.feedback_sub = self.node.create_subscription(
            InteractiveMarkerFeedback,
            feedback_topic,
            self.processFeedback,
            feedback_sub_qos
        )

    def shutdown(self):
        """
        Shutdown the interactive marker server.

        This should be called before the node is destroyed so that the internal ROS entities
        can be destroyed.
        """
        # It only makes sense to publish changes if the ROS context is still valid
        if self.node.context.ok():
            self.clear()
            self.applyChanges()
        if self.get_interactive_markers_srv is not None:
            self.node.destroy_service(self.get_interactive_markers_srv)
            self.get_interactive_markers_srv = None
        if self.update_pub is not None:
            self.node.destroy_publisher(self.update_pub)
            self.update_pub = None
        if self.feedback_sub is not None:
            self.node.destroy_subscription(self.feedback_sub)
            self.feedback_sub = None

    def __del__(self):
        """Destruction of the interface will lead to all managed markers being cleared."""
        self.shutdown()

    def insert(self, marker, *, feedback_callback=None, feedback_type=DEFAULT_FEEDBACK_CALLBACK):
        """
        Add or replace a marker.

        Note: Changes to the marker will not take effect until you call applyChanges().
        The callback changes immediately.

        :param marker: The marker to be added or replaced.
        :param feedback_callback: Function to call on the arrival of a feedback message.
        :param feedback_type: Type of feedback for which to call the feedback.
        """
        with self.mutex:
            if marker.name in self.pending_updates:
                update = self.pending_updates[marker.name]
            else:
                update = UpdateContext()
                self.pending_updates[marker.name] = update
            update.update_type = UpdateContext.FULL_UPDATE
            update.int_marker = marker
        if feedback_callback is not None:
            self.setCallback(marker.name, feedback_callback, feedback_type)

    def setPose(self, name, pose, header=Header()):
        """
        Update the pose of a marker with the specified name.

        Note: This change will not take effect until you call applyChanges()

        :param name: Name of the interactive marker.
        :param pose: The new pose.
        :param header: Header replacement. Leave this empty to use the previous one.
        :return: True if a marker with that name exists, False otherwise.
        """
        with self.mutex:
            marker_context = self.marker_contexts.get(name, None)
            update = self.pending_updates.get(name, None)
            # if there's no marker and no pending addition for it, we can't update the pose
            if marker_context is None and update is None:
                return False
            if update is not None and update.update_type == UpdateContext.FULL_UPDATE:
                return False

            if header.frame_id is None or header.frame_id == '':
                # keep the old header
                self.doSetPose(update, name, pose, marker_context.int_marker.header)
            else:
                self.doSetPose(update, name, pose, header)
            return True

    def erase(self, name):
        """
        Erase the marker with the specified name.

        Note: This change will not take effect until you call applyChanges().

        :param name: Name of the interactive marker.
        :return: True if a marker with that name exists, False otherwise.
        """
        with self.mutex:
            if name in self.pending_updates:
                self.pending_updates[name].update_type = UpdateContext.ERASE
                return True
            if name in self.marker_contexts:
                update = UpdateContext()
                update.update_type = UpdateContext.ERASE
                self.pending_updates[name] = update
                return True
            return False

    def clear(self):
        """
        Clear all markers.

        Note: This change will not take effect until you call applyChanges().
        """
        self.pending_updates = {}
        for marker_name in self.marker_contexts.keys():
            self.erase(marker_name)

    def setCallback(self, name, feedback_callback, feedback_type=DEFAULT_FEEDBACK_CALLBACK):
        """
        Add or replace a callback function for the specified marker.

        Note: This change will not take effect until you call applyChanges().
        The server will try to call any type-specific callback first.
        If a callback for the given type already exists, it will be replaced.
        To unset a callback, pass a value of None.

        :param name: Name of the interactive marker
        :param feedback_callback: Function to call on the arrival of a feedback message.
        :param feedback_type: Type of feedback for which to call the feedback.
            Leave this empty to make this the default callback.
        """
        with self.mutex:
            marker_context = self.marker_contexts.get(name, None)
            update = self.pending_updates.get(name, None)
            if marker_context is None and update is None:
                return False

            # we need to overwrite both the callbacks for the actual marker
            # and the update, if there's any
            if marker_context:
                # the marker exists, so we can just overwrite the existing callbacks
                if feedback_type == self.DEFAULT_FEEDBACK_CALLBACK:
                    marker_context.default_feedback_callback = feedback_callback
                else:
                    if feedback_callback:
                        marker_context.feedback_callbacks[feedback_type] = feedback_callback
                    elif feedback_type in marker_context.feedback_callbacks:
                        del marker_context.feedback_callbacks[feedback_type]
            if update:
                if feedback_type == self.DEFAULT_FEEDBACK_CALLBACK:
                    update.default_feedback_callback = feedback_callback
                else:
                    if feedback_callback:
                        update.feedback_callbacks[feedback_type] = feedback_callback
                    elif feedback_type in update.feedback_callbacks:
                        del update.feedback_callbacks[feedback_type]
            return True

    def applyChanges(self):
        """Apply changes made since the last call to this method and broadcast to clients."""
        with self.mutex:
            if len(self.pending_updates.keys()) == 0:
                return

            update_msg = InteractiveMarkerUpdate()
            update_msg.type = InteractiveMarkerUpdate.UPDATE

            for name, update in self.pending_updates.items():
                if update.update_type == UpdateContext.FULL_UPDATE:
                    if name in self.marker_contexts:
                        marker_context = self.marker_contexts[name]
                    else:
                        self.node.get_logger().debug('Creating new context for ' + name)
                        # create a new int_marker context
                        marker_context = MarkerContext(self.node.get_clock().now())
                        marker_context.default_feedback_callback = update.default_feedback_callback
                        marker_context.feedback_callbacks = update.feedback_callbacks
                        self.marker_contexts[name] = marker_context

                    marker_context.int_marker = update.int_marker
                    update_msg.markers.append(marker_context.int_marker)

                elif update.update_type == UpdateContext.POSE_UPDATE:
                    if name not in self.marker_contexts:
                        self.node.get_logger().error(
                            'Pending pose update for non-existing marker found. '
                            'This is a bug in InteractiveMarkerServer.')
                        continue

                    marker_context = self.marker_contexts[name]
                    marker_context.int_marker.pose = update.int_marker.pose
                    marker_context.int_marker.header = update.int_marker.header

                    pose_update = InteractiveMarkerPose()
                    pose_update.header = marker_context.int_marker.header
                    pose_update.pose = marker_context.int_marker.pose
                    pose_update.name = marker_context.int_marker.name
                    update_msg.poses.append(pose_update)

                elif update.update_type == UpdateContext.ERASE:
                    if name in self.marker_contexts:
                        marker_context = self.marker_contexts[name]
                        del self.marker_contexts[name]
                        update_msg.erases.append(name)
            self.pending_updates = {}

        self.seq_num += 1
        self.publish(update_msg)

    def get(self, name):
        """
        Get marker by name.

        :param name: Name of the interactive marker.
        :return: Marker if exists, None otherwise.
        """
        if name in self.pending_updates:
            update = self.pending_updates[name]
        elif name in self.marker_contexts:
            return self.marker_contexts[name].int_marker
        else:
            return None

        # if there's an update pending, we'll have to account for that
        if update.update_type == UpdateContext.ERASE:
            return None
        elif update.update_type == UpdateContext.POSE_UPDATE:
            if name not in self.marker_contexts:
                return None
            marker_context = self.marker_contexts[name]
            int_marker = marker_context.int_marker
            int_marker.pose = update.int_marker.pose
            return int_marker
        elif update.update_type == UpdateContext.FULL_UPDATE:
            return update.int_marker
        return None

    def processFeedback(self, feedback):
        """Update marker pose and call user callback."""
        with self.mutex:
            # ignore feedback for non-existing markers
            if feedback.marker_name not in self.marker_contexts:
                return

            marker_context = self.marker_contexts[feedback.marker_name]

            # if two callers try to modify the same marker, reject (timeout= 1 sec)
            time_since_last_feedback = self.node.get_clock().now() - marker_context.last_feedback
            if (marker_context.last_client_id != feedback.client_id and
                    time_since_last_feedback < Duration(seconds=1.0)):
                self.node.get_logger().debug(
                    "Rejecting feedback for '{}': conflicting feedback from separate clients"
                    .format(feedback.marker_name)
                )
                return

            marker_context.last_feedback = self.node.get_clock().now()
            marker_context.last_client_id = feedback.client_id

            if feedback.event_type == feedback.POSE_UPDATE:
                if (marker_context.int_marker.header.stamp == Time()):
                    # keep the old header
                    header = marker_context.int_marker.header
                else:
                    header = feedback.header

                if feedback.marker_name in self.pending_updates:
                    self.doSetPose(
                        self.pending_updates[feedback.marker_name],
                        feedback.marker_name,
                        feedback.pose,
                        header
                    )
                else:
                    self.doSetPose(None, feedback.marker_name, feedback.pose, header)

        # call feedback handler
        feedback_callback = marker_context.feedback_callbacks.get(
            feedback.event_type, marker_context.default_feedback_callback)
        if feedback_callback is not None:
            feedback_callback(feedback)

        # apply any pose updates
        self.applyChanges()

    def publish(self, update):
        """Increase the sequence number and publish an update."""
        update.seq_num = self.seq_num
        self.update_pub.publish(update)

    def getInteractiveMarkersCallback(self, request, response):
        """Process a service request to get the current interactive markers."""
        with self.mutex:
            response.sequence_number = self.seq_num

            self.node.get_logger().debug(
                'Markers requested. Responding with the following markers:'
            )
            for name, marker_context in self.marker_contexts.items():
                self.node.get_logger().debug('    ' + name)
                response.markers.append(marker_context.int_marker)

            return response

    def doSetPose(self, update, name, pose, header):
        """Schedule a pose update pose without locking."""
        if update is None:
            update = UpdateContext()
            update.update_type = UpdateContext.POSE_UPDATE
            self.pending_updates[name] = update
        elif update.update_type != UpdateContext.FULL_UPDATE:
            update.update_type = UpdateContext.POSE_UPDATE

        update.int_marker.pose = pose
        update.int_marker.header = header


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from visualization_msgs.msg import InteractiveMarkerFeedback
from std_msgs.msg import Header, ColorRGBA

import math
import csv
import os
from enum import Enum


class MarkerMode(Enum):
    """Mode for marker initialization and saving."""
    SAVE = "save"
    LOAD = "load"


class InteractiveWaypointNode(Node):
    """
    Interactive Marker 기반 Waypoint 편집 노드.

    - N개의 Interactive Marker(실린더, r=0.3m, h=0.1m)를 생성
    - 각 마커는 정수형 id(ns 기반)를 가짐
    - id 기준 소팅하여 PoseArray를 발행
    """

    def __init__(self):
        super().__init__('interactive_waypoint_node')

        # ── 파라미터 ──

        self.num_markers = 10 # Number of markers
        self.frame_id = "map"
        self.marker_radius = 0.3 # Marker radius
        self.marker_height = 0.1 # Marker height
        self.publish_rate = 5.0 # Hz

        # ── Mode 설정 (SAVE or LOAD) ──        
        self.mode = MarkerMode.SAVE # MarkerMode.LOAD
        self.csv_path: str = None

        # ── Interactive Marker Server ──
        self.server = InteractiveMarkerServer(self, 'waypoint_markers')

        # ── PoseArray Publisher ──
        self.pose_array_pub = self.create_publisher(PoseArray, 'ring_poses', 10)

        # ── 마커별 현재 포즈 저장 (id -> Pose) ──
        self.marker_poses: dict[int, Pose] = {}

        # ── 마커 생성 ──
        self._create_markers()

        # ── 주기적 PoseArray 발행 ──
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self._publish_pose_array)

        self.get_logger().info(
            f'InteractiveWaypointNode started with {self.num_markers} markers '
            f'(r={self.marker_radius}, h={self.marker_height}), mode={self.mode.value}'
        )

    # ================================================================
    #  마커 생성
    # ================================================================
    def _create_markers(self):
        """num_markers 개의 Interactive Marker를 일렬로 배치하여 생성."""
        # LOAD 모드이거나 SAVE 모드에서 csv_path가 제공된 경우 CSV에서 로드
        if self.mode == MarkerMode.LOAD or (self.mode == MarkerMode.SAVE and self.csv_path):
            if not self.csv_path:
                self.get_logger().error('LOAD mode requires csv_path parameter!')
                return
            
            if not os.path.exists(self.csv_path):
                self.get_logger().error(f'CSV file not found: {self.csv_path}')
                return
            
            loaded_poses = self._load_poses_from_csv(self.csv_path)
            if not loaded_poses:
                self.get_logger().warn(f'No poses loaded from {self.csv_path}, using default positions')
                self._create_default_markers()
            else:
                self._create_markers_from_poses(loaded_poses)
                self.get_logger().info(f'Loaded {len(loaded_poses)} markers from {self.csv_path}')
        else:
            # SAVE 모드에서 csv_path가 없으면 기본 위치로 생성
            self._create_default_markers()

        self.server.applyChanges()

    def _create_default_markers(self):
        """기본 위치(x축 1m 간격)로 마커 생성."""
        for i in range(self.num_markers):
            marker_id = i
            name = f'waypoint_{marker_id}'

            # 초기 위치: x 축 방향으로 1m 간격
            init_pose = Pose()
            init_pose.position.x = float(i) * 1.0
            init_pose.position.y = 0.0
            init_pose.position.z = 0.0
            init_pose.orientation.w = 1.0

            int_marker = self._make_interactive_marker(name, marker_id, init_pose)
            self.server.insert(int_marker, feedback_callback=self._feedback_callback)

            # 초기 포즈 저장
            self.marker_poses[marker_id] = init_pose

    def _create_markers_from_poses(self, poses: list[Pose]):
        """주어진 Pose 리스트로부터 마커 생성."""
        for i, pose in enumerate(poses):
            marker_id = i
            name = f'waypoint_{marker_id}'

            int_marker = self._make_interactive_marker(name, marker_id, pose)
            self.server.insert(int_marker, feedback_callback=self._feedback_callback)

            # 포즈 저장
            self.marker_poses[marker_id] = pose

        # num_markers 업데이트
        self.num_markers = len(poses)

    # ================================================================
    #  CSV 저장/로드
    # ================================================================
    def _load_poses_from_csv(self, csv_path: str) -> list[Pose]:
        """CSV 파일에서 Pose 리스트를 로드."""
        poses = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pose = Pose()
                    pose.position.x = float(row['x'])
                    pose.position.y = float(row['y'])
                    pose.position.z = float(row['z'])
                    pose.orientation.x = float(row['qx'])
                    pose.orientation.y = float(row['qy'])
                    pose.orientation.z = float(row['qz'])
                    pose.orientation.w = float(row['qw'])
                    poses.append(pose)
        except Exception as e:
            self.get_logger().error(f'Failed to load CSV: {e}')
            return []
        
        return poses

    def _save_poses_to_csv(self, csv_path: str):
        """현재 마커 포즈들을 CSV 파일로 저장."""
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['id', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # id 기준 정렬하여 저장
                for marker_id in sorted(self.marker_poses.keys()):
                    pose = self.marker_poses[marker_id]
                    writer.writerow({
                        'id': marker_id,
                        'x': pose.position.x,
                        'y': pose.position.y,
                        'z': pose.position.z,
                        'qx': pose.orientation.x,
                        'qy': pose.orientation.y,
                        'qz': pose.orientation.z,
                        'qw': pose.orientation.w
                    })
            
            self.get_logger().info(f'Saved {len(self.marker_poses)} marker poses to {csv_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')

    def _make_interactive_marker(
        self, name: str, marker_id: int, pose: Pose
    ) -> InteractiveMarker:
        """실린더 형태의 6-DOF Interactive Marker를 생성."""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame_id
        int_marker.name = name
        int_marker.description = f'WP {marker_id}'
        int_marker.pose = pose
        int_marker.scale = max(self.marker_radius * 3.0, 0.45)

        # ── 시각화용 실린더 마커 ──
        cylinder_marker = Marker()
        cylinder_marker.type = Marker.CYLINDER
        cylinder_marker.scale.x = self.marker_radius * 2.0  # diameter
        cylinder_marker.scale.y = self.marker_radius * 2.0
        cylinder_marker.scale.z = self.marker_height
        # id 에 따라 색상을 다르게 (hue 변환)
        color = self._id_to_color(marker_id)
        cylinder_marker.color = color
        
        # 실린더를 x축 방향으로 90도 회전 (시각적 표현만)
        cylinder_marker.pose.orientation.x = 0.0
        cylinder_marker.pose.orientation.y = math.sqrt(2.0) / 2.0
        cylinder_marker.pose.orientation.z = 0.0
        cylinder_marker.pose.orientation.w = math.sqrt(2.0) / 2.0

        # 실린더 표시 컨트롤
        visual_control = InteractiveMarkerControl()
        visual_control.always_visible = True
        visual_control.markers.append(cylinder_marker)
        visual_control.interaction_mode = InteractiveMarkerControl.NONE
        int_marker.controls.append(visual_control)

        # ── 6-DOF 이동/회전 컨트롤 ──
        # X-Y 평면 이동 (가장 많이 사용)
        move_xy = InteractiveMarkerControl()
        move_xy.name = 'move_xy'
        move_xy.orientation.w = 1.0
        move_xy.orientation.x = 0.0
        move_xy.orientation.y = 1.0
        move_xy.orientation.z = 0.0
        move_xy.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        move_xy.always_visible = False
        int_marker.controls.append(move_xy)

        # X축 이동
        control_x = InteractiveMarkerControl()
        control_x.name = 'move_x'
        control_x.orientation.w = 1.0
        control_x.orientation.x = 1.0
        control_x.orientation.y = 0.0
        control_x.orientation.z = 0.0
        control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_x)

        # Y축 이동
        control_y = InteractiveMarkerControl()
        control_y.name = 'move_y'
        control_y.orientation.w = 1.0
        control_y.orientation.x = 0.0
        control_y.orientation.y = 0.0
        control_y.orientation.z = 1.0
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_y)

        # Z축 이동
        control_z = InteractiveMarkerControl()
        control_z.name = 'move_z'
        control_z.orientation.w = 1.0
        control_z.orientation.x = 0.0
        control_z.orientation.y = 1.0
        control_z.orientation.z = 0.0
        control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_z)

        # X축 회전 (Pitch)
        control_pitch = InteractiveMarkerControl()
        control_pitch.name = 'rotate_x'
        control_pitch.orientation.w = 1.0
        control_pitch.orientation.x = 1.0
        control_pitch.orientation.y = 0.0
        control_pitch.orientation.z = 0.0
        control_pitch.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_pitch)

        # Y축 회전 (Roll)
        control_roll = InteractiveMarkerControl()
        control_roll.name = 'rotate_y'
        control_roll.orientation.w = 1.0
        control_roll.orientation.x = 0.0
        control_roll.orientation.y = 0.0
        control_roll.orientation.z = 1.0
        control_roll.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_roll)
        
        # Z축 회전 (Yaw)
        control_yaw = InteractiveMarkerControl()
        control_yaw.name = 'rotate_z'
        control_yaw.orientation.w = 1.0
        control_yaw.orientation.x = 0.0
        control_yaw.orientation.y = 1.0
        control_yaw.orientation.z = 0.0
        control_yaw.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(control_yaw)

        return int_marker

    # ================================================================
    #  콜백
    # ================================================================
    def _feedback_callback(self, feedback: InteractiveMarkerFeedback):
        """마커 이동 시 포즈를 업데이트."""
        # 이름에서 id 추출: 'waypoint_3' -> 3
        marker_id = self._name_to_id(feedback.marker_name)
        if marker_id is None:
            return

        self.marker_poses[marker_id] = feedback.pose

        self.get_logger().debug(
            f'Marker {marker_id} moved to '
            f'({feedback.pose.position.x:.2f}, '
            f'{feedback.pose.position.y:.2f}, '
            f'{feedback.pose.position.z:.2f})'
        )

    # ================================================================
    #  PoseArray 발행 (id 기준 소팅)
    # ================================================================
    def _publish_pose_array(self):
        """marker_poses를 정수 id 기준으로 소팅하여 PoseArray 발행."""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.frame_id

        # 정수 id 기준 오름차순 소팅
        for marker_id in sorted(self.marker_poses.keys()):
            pose_array.poses.append(self.marker_poses[marker_id])

        self.pose_array_pub.publish(pose_array)

    # ================================================================
    #  유틸
    # ================================================================
    @staticmethod
    def _name_to_id(name: str) -> int | None:
        """'waypoint_<id>' 형식 이름에서 정수 id를 추출."""
        try:
            return int(name.split('_')[-1])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _id_to_color(marker_id: int, total: int = 10) -> ColorRGBA:
        """id에 따라 HSV hue를 변경하여 구분 가능한 색상 반환."""
        hue = (marker_id * 137.508) % 360.0  # golden angle for good distribution
        return InteractiveWaypointNode._hsv_to_rgba(hue, 0.8, 0.9, 0.8)

    @staticmethod
    def _hsv_to_rgba(h: float, s: float, v: float, a: float = 1.0) -> ColorRGBA:
        """HSV -> RGBA 변환."""
        c = v * s
        x = c * (1.0 - abs((h / 60.0) % 2 - 1.0))
        m = v - c

        if h < 60:
            r, g, b = c, x, 0.0
        elif h < 120:
            r, g, b = x, c, 0.0
        elif h < 180:
            r, g, b = 0.0, c, x
        elif h < 240:
            r, g, b = 0.0, x, c
        elif h < 300:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x

        color = ColorRGBA()
        color.r = r + m
        color.g = g + m
        color.b = b + m
        color.a = a
        return color

    def destroy_node(self):
        # SAVE 모드일 경우 종료 시 CSV 저장
        if self.mode == MarkerMode.SAVE:
            if not self.csv_path:
                # 기본 경로 사용
                self.csv_path = os.path.join(os.path.expanduser('~'), 'waypoint_markers.csv')
            self._save_poses_to_csv(self.csv_path)
        
        self.server.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveWaypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()