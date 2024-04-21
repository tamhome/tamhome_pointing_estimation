#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from audioop import cross
import roslib
import rospy
import tf
import numpy as np
import sys
import time
import json
import os
import math
import random
import threading
import collections

from tamlib.utils import Logger


from std_msgs.msg import Bool
from geometry_msgs.msg import Point, PointStamped
from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from visualization_msgs.msg import Marker, MarkerArray
from tam_mmaction2.msg import Ax3DPoseWithLabel, Ax3DPoseWithLabelArray
from tam_dynamic_map.srv import GetAllObjectPose
from tam_dynamic_map.srv import GetAllObjectPoseResponse

class PointingEstimation(Logger):
    def __init__(self):
        Logger.__init__(self, loglevel="INFO")

        self.pose_threshold = rospy.get_param("~pose_th", 0.4)
        self.pointing_threshold = rospy.get_param("~pointing_th", 0.2)
        self.elbow_distance_threshold = rospy.get_param("~elbow_distance_th", 0.07)
        self.buffer_len = rospy.get_param("~buffer_length/pose", 10)
        self.pointing_buffer_len = rospy.get_param("~buffer_length/pointing", 30)
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.run_enable = rospy.get_param("~auto_start", True)  # Falseに変更予定
        self.use_ic = rospy.get_param("~use_ic", True)

        self.tf_listener = tf.TransformListener()

        # 制御用変数
        self.latest_pose = None
        self.update_flag = False
        self.prv_pointed_furniture = None  # どの家具を指差していたかを制御
        self.right_arm_pose_buffer = collections.deque(maxlen=self.buffer_len)
        self.left_arm_pose_buffer = collections.deque(maxlen=self.buffer_len)
        self.pointing_position_buffer = collections.deque(maxlen=self.pointing_buffer_len)

        # ros interface
        rospy.Subscriber("/mmaction2/poses/with_label", Ax3DPoseWithLabelArray, self.cb_pose_array, queue_size=1)
        if self.use_ic:
            from interactive_cleanup.msg import InteractiveCleanupMsg
            rospy.Subscriber("/interactive_cleanup/message/to_robot", InteractiveCleanupMsg, self.cb_moderator_info, queue_size=1)
            self.pub_to_moderator = rospy.Publisher("/interactive_cleanup/message/to_moderator", InteractiveCleanupMsg)
            self.focus_start_time = rospy.Time.now()
            self.focus_duration = rospy.Duration(1)  # メッセージが来てから注目する時間

            self.start_estimation_time = rospy.Time.now()
            self.end_pointing_time = None
            self.repointing_timeout = rospy.Duration(70)  # メッセージが来てから注目する時間

            self.to_robot_msg = None

            self.flag_pickup_set = False
            self.flag_cleanup_set = False
            rospy.set_param("/interactive_cleanup/pickup/point", 0)
            rospy.set_param("/interactive_cleanup/cleanup/point", 0)
            rospy.set_param("/interactive_cleanup/task/start", False)

        self.pub_pointing_line = rospy.Publisher("~pointing_line", Marker, queue_size=1)
        self.pub_pointing_position = rospy.Publisher("~pointing_position", Marker, queue_size=1)
        self.srv_get_world_model = rospy.ServiceProxy("/tam_dynamic_map/get_all_obj_pose/service", GetAllObjectPose)
        self.srv_focus_person_enable = rospy.Service("/pointing_estimation/run_enable", SetBool, self.cb_run_enable)

    def delete(self):
        """デストラクタ
        """
        return

    def cb_run_enable(self, req: SetBoolRequest) -> SetBoolResponse:
        response = SetBoolResponse()
        if req.data:
            self.run_enable = True
            self.loginfo("start pointing estimation node!")
            response.success = True
            response.message = "start"
        else:
            self.run_enable = False
            self.loginfo("stop pointing estimation node!")
            response.success = True
            response.message = "stop"

        return response

    # コールバック関数
    def cb_pose_array(self, msg) -> None:
        """最新の骨格情報を取得する関数
        """
        self.latest_pose = msg
        self.base_frame = msg.header.frame_id
        self.update_flag = True

    def cb_moderator_info(self, msg) -> None:
        """Interactive Cleanupのメッセージを取得
        """
        self.to_robot_msg = msg.message
        if self.to_robot_msg == "Pick_it_up!":
            self.focus_start_time = rospy.Time.now()
        if self.to_robot_msg == "Clean_up!":
            self.focus_start_time = rospy.Time.now()
            self.end_pointing_time = rospy.Time.now()

    def plz_point_again(self) -> None:
        """制御用変数の初期化と再度の指差し要求を行う関数
        """
        self.loginfo("pointing estimation is failed. Please point it aggain")
        self.flag_pickup_set = False
        self.flag_cleanup_set = False
        self.start_estimation_time = rospy.Time.now()
        self.end_pointing_time = None

        rospy.set_param("/interactive_cleanup/pickup/point", 0)
        rospy.set_param("/interactive_cleanup/cleanup/point", 0)
        rospy.set_param("/interactive_cleanup/task/start", False)

        msg = InteractiveCleanupMsg(message="Point_it_again")
        self.pub_to_moderator.publish(msg)

    def check_straight_arm(self, target_arm_position) -> bool:
        """腕が伸びているかを確認する関数（伸びている＝指を指している）
        """
        center_position = (target_arm_position[0] + target_arm_position[2]) / 2.0
        disntance_elbow_to_center = np.linalg.norm(target_arm_position[1] - center_position)
        # print(disntance_elbow_to_center)
        if disntance_elbow_to_center > self.elbow_distance_threshold:
            return False
        else:
            return True

    def check_cross_plane(self, target_wrist_point, target_direction_vector, furniture):
        """平面と指差しラインの直行しているポイントを見つける
        """

        center_pose = furniture.pose.position
        scale = furniture.scale

        plane_center_point = [center_pose.x, center_pose.y, center_pose.z]
        plane = [
            [center_pose.x - ((scale[0]) / 2), center_pose.y - ((scale[1]) / 2), center_pose.z],
            [center_pose.x + ((scale[0]) / 2), center_pose.y + ((scale[1]) / 2), center_pose.z]
        ]
        np_plane = np.array(plane)
        self.logdebug(np_plane)

        # check which axis is horizontal
        axis = None
        if np_plane[:, 0].max() == np_plane[:, 0].min():
            axis = 0
        elif np_plane[:, 1].max() == np_plane[:, 1].min():
            axis = 1
        elif np_plane[:, 2].max() == np_plane[:, 2].min():
            axis = 2
        else:
            rospy.logwarn("Invalid axis.")
            return None
        self.logdebug(f"axis is {axis}")

        param_t = (np_plane[0][axis] - target_wrist_point[axis]) / target_direction_vector[axis]
        if param_t <= 0.0 or param_t >= 10.0:
            return None

        cross_point = target_wrist_point + param_t * target_direction_vector
        self.logdebug(cross_point)

        # chech inside or outside
        if axis == 0 and \
        abs(cross_point[1] - plane_center_point[1]) <= abs(np_plane[:, 1].max() - np_plane[:, 1].min()) / 2.0 and \
        abs(cross_point[2] - plane_center_point[2]) <= abs(np_plane[:, 2].max() - np_plane[:, 2].min()) / 2.0:
            return cross_point
        elif axis == 1 and \
        abs(cross_point[0] - plane_center_point[0]) <= abs(np_plane[:, 0].max() - np_plane[:, 0].min()) / 2.0 and \
        abs(cross_point[2] - plane_center_point[2]) <= abs(np_plane[:, 2].max() - np_plane[:, 2].min()) / 2.0:
            return cross_point
        elif axis == 2 and \
        abs(cross_point[0] - plane_center_point[0]) <= abs(np_plane[:, 0].max() - np_plane[:, 0].min()) / 2.0 and \
        abs(cross_point[1] - plane_center_point[1]) <= abs(np_plane[:, 1].max() - np_plane[:, 1].min()) / 2.0:
            return cross_point

        else:
            return None

    def display_pointing_line(self, point_base, point_target, target_frame="map") -> None:
        """指差しの軌跡を描画
        """
        marker = Marker()
        marker.header.frame_id = target_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pointing_line"
        marker.lifetime = rospy.Duration(0.3)
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.color.r = random.random()
        marker.color.b = random.random()
        marker.color.g = random.random()
        marker.color.a = 1.0
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.points.append(Point(point_base[0], point_base[1], point_base[2]))
        marker.points.append(Point(point_target[0], point_target[1], point_target[2]))
        self.pub_pointing_line.publish(marker)

    def display_pointing_position(self, crosss_point, target_frame="map", lifetime=0.1) -> None:
        """家具の平面と一致したポイントを描画
        """
        marker = Marker()
        marker.header.frame_id = target_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pointing_position"
        marker.lifetime = rospy.Duration(lifetime)
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.color.r = 1.0
        marker.color.b = 0.0
        marker.color.g = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.pose.position = Point(crosss_point[0], crosss_point[1], crosss_point[2])
        self.pub_pointing_position.publish(marker)

    def transform_point(self, target_frame: str, base_frame: str, point):
        """
        """
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.tf_listener.lookupTransform(target_frame, base_frame, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.logwarn(e)
                self.logwarn("cannot calculate tf")
                continue
        rot_matrix = tf.transformations.quaternion_matrix(rot)
        trans_matrix = tf.transformations.translation_matrix(trans)
        plane_vec = np.array([point[0], point[1], point[2], 1.0])
        transform_matrix = np.dot(trans_matrix, rot_matrix)
        transed_plane_vec = np.dot(transform_matrix, plane_vec)

        return np.array([transed_plane_vec[0], transed_plane_vec[1], transed_plane_vec[2]])

    def run(self):
        """指差し認識の動作関数
        """
        while not rospy.is_shutdown():

            if not self.run_enable:
                rospy.sleep(0.1)
                continue

            # 新しい姿勢推定情報が入ってきていない場合はスキップ
            if not self.update_flag:
                self.logdebug("update flag is False")
                rospy.sleep(0.1)
                continue

            # 最新の人物の姿勢を取得
            try:
                target_people_array = self.latest_pose.people
                target_person = target_people_array[0]
            except AttributeError as e:
                self.logwarn(e)
                self.update_flag = False
                continue
            except IndexError as e:
                self.logdebug(e)
                self.logdebug("human did not detected")
                self.update_flag = False
                continue

            # 制御用の変数初期化
            left_arm_pose  = None
            right_arm_pose = None

            # アームの信頼値を参照し，使用するデータかどうかを決定
            # check arm pose is available
            if target_person.keypoints.left_shoulder.score > self.pose_threshold or target_person.keypoints.left_elbow.score > self.pose_threshold or target_person.keypoints.left_wrist.score > self.pose_threshold:
                left_arm_pose = np.array(
                    [
                        [target_person.keypoints.left_shoulder.point.x, target_person.keypoints.left_shoulder.point.y, target_person.keypoints.left_shoulder.point.z],
                        [target_person.keypoints.left_elbow.point.x, target_person.keypoints.left_elbow.point.y, target_person.keypoints.left_elbow.point.z],
                        [target_person.keypoints.left_wrist.point.x, target_person.keypoints.left_wrist.point.y, target_person.keypoints.left_wrist.point.z]
                    ]
                )
                self.left_arm_pose_buffer.append(left_arm_pose)

            if target_person.keypoints.right_shoulder.score > self.pose_threshold or target_person.keypoints.right_elbow.score > self.pose_threshold or target_person.keypoints.right_wrist.score > self.pose_threshold:
                right_arm_pose = np.array(
                    [
                        [target_person.keypoints.right_shoulder.point.x, target_person.keypoints.right_shoulder.point.y, target_person.keypoints.right_shoulder.point.z],
                        [target_person.keypoints.right_elbow.point.x, target_person.keypoints.right_elbow.point.y, target_person.keypoints.right_elbow.point.z],
                        [target_person.keypoints.right_wrist.point.x, target_person.keypoints.right_wrist.point.y, target_person.keypoints.right_wrist.point.z]
                    ]
                )
                self.right_arm_pose_buffer.append(right_arm_pose)

            # どちらの腕で指差しを行っているのかを検証
            # If neither arm is available, skip
            if left_arm_pose is None and right_arm_pose is None:
                self.logwarn("腕が画角内に写っていません．")
                self.update_flag = False
                continue

            # one arm is available
            elif left_arm_pose is None:
                self.loginfo("Focus on right arm")
                if len(self.right_arm_pose_buffer) <= self.buffer_len:
                    # target_positon_median = np.median(self.right_arm_pose_buffer.pop(), axis=0)
                    target_positon_median = self.right_arm_pose_buffer.pop()
                    target_wrist_point = target_positon_median[2]
                    target_shoulder_point = target_positon_median[0]
                else:
                    # self._lock.release()
                    self.update_flag = False
                    continue
            elif right_arm_pose is None:
                self.loginfo("Focus on left arm")
                if len(self.left_arm_pose_buffer) <= self.buffer_len:
                    # target_positon_median = np.median(self.left_arm_pose_buffer.pop(), axis=0)
                    target_positon_median = self.left_arm_pose_buffer.pop()
                    target_wrist_point = target_positon_median[2]
                    target_shoulder_point = target_positon_median[0]
                else:
                    # self._lock.release()
                    self.update_flag = False
                    continue

            # if both arm is available
            else:
                left_arm_direction_vector = left_arm_pose[2] - left_arm_pose[0]
                right_arm_direction_vector = right_arm_pose[2] - right_arm_pose[0]
                left_shoulder2elbow_direction_vector = left_arm_pose[1] - left_arm_pose[0]
                right_shoulder2elbow_direction_vector = right_arm_pose[1] - right_arm_pose[0]

                # どちらも指を指していない場合
                if abs(left_arm_direction_vector[0]) < self.pointing_threshold and abs(left_arm_direction_vector[2]) < self.pointing_threshold and abs(right_arm_direction_vector[0]) < self.pointing_threshold and abs(right_arm_direction_vector[2]) < self.pointing_threshold:
                    # self._lock.release()
                    self.update_flag = False
                    self.loginfo("Person do not pointing. skip")
                    continue

                # 左腕で指を指していた場合
                elif abs(left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) > abs(right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) and self.check_straight_arm(left_arm_pose):
                    self.logdebug("Focus on left arm")
                    if len(self.left_arm_pose_buffer) <= self.buffer_len:
                        self.logdebug("calc_pointing_line")
                        target_positon_median = self.left_arm_pose_buffer.pop()
                        # target_positon_median  = np.median(self.left_arm_pose_buffer.pop(), axis=0)
                        target_wrist_point = target_positon_median[2]
                        target_shoulder_point = target_positon_median[0]
                    else:
                        # self._lock.release()
                        self.logdebug("wait next message")
                        self.update_flag = False
                        # continue

                # 右腕で指を指していた場合
                elif abs(right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) > abs(left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) and self.check_straight_arm(right_arm_pose):
                    self.logdebug("Focus on right arm")
                    if len(self.right_arm_pose_buffer) <= self.buffer_len:
                        target_positon_median = self.right_arm_pose_buffer.pop()
                        target_wrist_point = target_positon_median[2]
                        target_shoulder_point = target_positon_median[0]
                    else:
                        # self._lock.release()
                        self.update_flag = False
                        # continue

                else:
                    # self._lock.release()
                    self.logdebug("Cannot detect pointing direction. skip")
                    self.update_flag = False
                    continue

            target_frame = "map"
            transed_target_wrist_point = self.transform_point(target_frame, self.base_frame, target_wrist_point)
            transed_target_shoulder_point = self.transform_point(target_frame, self.base_frame, target_shoulder_point)

            # debug display marker at target_frame
            self.logdebug("display pointing line")
            self.display_pointing_line(transed_target_wrist_point, transed_target_wrist_point + (transed_target_wrist_point - transed_target_shoulder_point) * 3.0, target_frame=target_frame)

            # check cross plane
            self.world_model = self.srv_get_world_model()
            for furniture_info in self.world_model.world_model.obj_poses:
                self.logdebug(f"info is: \n {furniture_info}")
                if furniture_info.id == "floor":
                    cross_point: np.array = self.check_cross_plane(transed_target_wrist_point, (transed_target_wrist_point - transed_target_shoulder_point), furniture=furniture_info)
                    if cross_point is not None:
                        self.display_pointing_position(cross_point, target_frame)
                        self.pointing_position_buffer.append(cross_point)
                        self.loginfo(cross_point)

                        # Interactive cleanupに情報を送信
                        if self.use_ic:
                            cross_point_list = cross_point.tolist()
                            current_time = rospy.Time.now()
                            elapsed_time = current_time - self.focus_start_time

                            # to_robot_msgの更新をもとに，把持座標と配置座標をparamに記録
                            if self.to_robot_msg == "Pick_it_up!" and elapsed_time < self.focus_duration:
                                self.flag_pickup_set = True
                                rospy.set_param("/interactive_cleanup/pickup/point", cross_point_list)

                            if self.to_robot_msg == "Clean_up!" and elapsed_time < self.focus_duration:
                                self.flag_cleanup_set = True
                                rospy.set_param("/interactive_cleanup/cleanup/point", cross_point_list)

            if self.use_ic:
                if self.flag_pickup_set is True:
                    if self.flag_cleanup_set is True:
                        # 両方準備が整った段階でタスク開始
                        self.logsuccess("pointing estimation is complete. task start!")
                        rospy.set_param("/interactive_cleanup/task/start", True)

                    # 指差しが終わったとき
                    if self.end_pointing_time is not None:
                        # 把持位置だけわかっていたら，タスク開始
                        self.loginfo("pointing estimation is done only pointing object. task start!")
                        rospy.set_param("/interactive_cleanup/task/start", True)

                elif self.flag_cleanup_set is True:
                    # 片付け位置だけわかった場合は，再度指差しを要求する
                    self.plz_point_again()

                # どちらもわからなかったとき
                else:
                    self.loginfo("could not estimate both pointing position")
                    if self.end_pointing_time is not None:
                        current_time = rospy.Time.now()
                        if (current_time - self.start_estimation_time) > rospy.Duration(5):
                            self.plz_point_again()


if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])
    cls = PointingEstimation()

    p_loop_rate = rospy.get_param("~loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        try:
            cls.run()
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        loop_wait.sleep()
