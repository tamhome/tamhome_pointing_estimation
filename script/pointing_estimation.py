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

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse


class PointingEstimation(Logger):
    def __init__(self):
        Logger.__init__(self)

        self.pose_threhold = rospy.get_param("~pose_th", 0.4)
        self.buffer_len = rospy.get_param("~buffer_length/pose", 10)
        self.pointing_buffer_len = rospy.get_param("~buffer_length/pointing", 30)
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.run_enable = rospy.get_param("~auto_start", True)  # Falseに変更予定

        # 制御用変数
        self.latest_pose = None
        self.prv_pointed_furniture = None  # どの家具を指差していたかを制御
        self.right_arm_pose_buffer = collections.deque(maxlen=self.buffer_len)
        self.left_arm_pose_buffer = collections.deque(maxlen=self.buffer_len)
        self.pointing_position_buffer = collections.deque(maxlen=self.pointing_buffer_len)

        # ros interface
        rospy.Subscriber("/mmaction2/pose/with_label", PoseWithLabel, self.cb_pose_array, queue_size=1)
        self.pub_pointing_line = rospy.Publisher("~pointing_line", Marker, queue_size=1)
        self.pub_pointing_position = rospy.Publisher("~pointing_position", Marker, queue_size=1)

    def delete(self):
        return

    # コールバック関数
    def cb_pose_array(self, msg) -> None:
        """最新の骨格情報を取得する関数
        """
        self.latest_pose = msg

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
        self._pub_pointing_line.publish(marker)

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
        self._pub_pointing_position.publish(marker)

    def run(self):
        """指差し認識の動作関数
        """
        while not rospy.is_shutdown():

            if not self.run_enable:
                rospy.sleep(0.1)
                continue

            # 最新の人物の姿勢を取得
            target_person = None

            # 制御用の変数初期化
            left_arm_pose  = None
            right_arm_pose = None

            # アームの信頼値を参照し，使用するデータかどうかを決定
            # check arm pose is available
            if target_person.shoulder_left.score > self.pose_threhold or target_person.elbow_left.score > self.pose_threhold or target_person.wrist_left.score > self.pose_threhold:
                left_arm_pose = np.array(
                    [
                        [target_person.shoulder_left.point.x, target_person.shoulder_left.point.y, target_person.shoulder_left.point.z],
                        [target_person.elbow_left.point.x, target_person.elbow_left.point.y, target_person.elbow_left.point.z],
                        [target_person.wrist_left.point.x, target_person.wrist_left.point.y, target_person.wrist_left.point.z]
                    ]
                )
                self.left_arm_pose_buffer.append(left_arm_pose)

            if target_person.shoulder_right.score > self.pose_threhold or target_person.elbow_right.score > self.pose_threhold or target_person.wrist_right.score > self.pose_threhold:
                right_arm_pose = np.array(
                    [
                        [target_person.shoulder_right.point.x, target_person.shoulder_right.point.y, target_person.shoulder_right.point.z],
                        [target_person.elbow_right.point.x, target_person.elbow_right.point.y, target_person.elbow_right.point.z],
                        [target_person.wrist_right.point.x, target_person.wrist_right.point.y, target_person.wrist_right.point.z]
                    ]
                )
                self.right_arm_pose_buffer.append(right_arm_pose)

            # どちらの腕で指差しを行っているのかを検証
            # If neither arm is available, skip
            if left_arm_pose is None and right_arm_pose is None:
                self.logwarn("腕が画角内に写っていません．")
                continue

            # one arm is available
            elif left_arm_pose is None:
                self.loginfo("Focus on right arm")
                if self.right_arm_pose_buffer.is_full:
                    target_positon_median = np.median(self.right_arm_pose_buffer.pop(), axis=0)
                    target_wrist_point = target_positon_median[2]
                    target_shoulder_point = target_positon_median[0]
                else:
                    self._lock.release()
                    continue
            elif right_arm_pose is None:
                self.loginfo("Focus on left arm")
                if self.left_arm_pose_buffer.is_full:
                    target_positon_median = np.median(self.left_arm_pose_buffer.pop(), axis=0)
                    target_wrist_point = target_positon_median[2]
                    target_shoulder_point = target_positon_median[0]
                else:
                    self._lock.release()
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
                    self.loginfo("Person do not pointing. skip")
                    continue

                # 左腕で指を指していた場合
                elif abs(left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) > abs(right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) and self.checkStraightArm(left_arm_pose):
                    self.logdebug("Focus on left arm")
                    if len(self.left_arm_pose_buffer) == self.buffer_len:
                        target_positon_median  = np.median(self.left_arm_pose_buffer.pop(), axis=0)
                        target_wrist_point     = target_positon_median[2]
                        target_shoulder_point  = target_positon_median[0]
                    else:
                        # self._lock.release()
                        continue

                # 右腕で指を指していた場合
                elif abs(right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) > abs(left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) and self.checkStraightArm(right_arm_pose):
                    self.logdebug("Focus on right arm")
                    if len(self.right_arm_pose_buffer) == self.buffer_len:
                        target_positon_median  = np.median(self.right_arm_pose_buffer.pop(), axis=0)
                        target_wrist_point     = target_positon_median[2]
                        target_shoulder_point  = target_positon_median[0]
                    else:
                        # self._lock.release()
                        continue

                else:
                    # self._lock.release()
                    self.logdebug("Cannot detect pointing direction. skip")
                    continue

            target_frame = self._furniture_json[0]["frame"]
            transed_target_wrist_point = self.transformPoint(target_frame, self.base_frame, target_wrist_point)
            transed_target_shoulder_point = self.transformPoint(target_frame, self.base_frame, target_shoulder_point)

            # debug display marker at target_frame
            self.display_pointing_line(transed_target_wrist_point, transed_target_wrist_point + (transed_target_wrist_point - transed_target_shoulder_point) * 3.0, target_frame=target_frame)

            # check cross plane
            for furniture in self._furniture_json:
                self._current_pointed_furniture = furniture["name"]
                self._current_height_th = furniture["height_th"]
                cross_point = self.checkCrossPlane(transed_target_wrist_point, (transed_target_wrist_point - transed_target_shoulder_point) , np.array( furniture["plane"]))
                if cross_point is not None:
                    self.display_pointing_position(cross_point, target_frame)
                    self.pointing_position_buffer.append(cross_point)
                    break

            # check pointed furniture change
            if self._current_pointed_furniture != self.prv_pointed_furniture and self.prv_pointed_furniture is not None:
                rospy.logwarn("Pointed Furniture Changed from {} to {}".format(self.prv_pointed_furniture, self._current_pointed_furniture))
                self.pointing_position_buffer.clear()
                self._current_pointed_furniture = self.prv_pointed_furniture = None

            if len(self.pointing_position_buffer) == self.pointing_buffer_len:
                pointed_position_median = np.median(self.pointing_position_buffer.pop(), axis=0)
                pointed_position_std = np.std(self.pointing_position_buffer.pop(), axis=0)
                print(pointed_position_median)
                print(pointed_position_std)
                if np.all(pointed_position_std < 1.0):
                    self.loginfo("Pointed furniture: {}".format(self._current_pointed_furniture))
                    self.loginfo("Pointed Position: {}".format(pointed_position_median))
                    rospy.set_param("~furniture", self._current_pointed_furniture)
                    rospy.set_param("~position", pointed_position_median.tolist())
                    rospy.set_param("~height_th", self._current_height_th)
                    self.display_pointing_position(pointed_position_median, target_frame, lifetime=0)
                    self.run_enable = False

            self.prv_pointed_furniture = self._current_pointed_furniture
            self._current_pointed_furniture = None


if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])
    cls = PointingEstimation()

    p_loop_rate = rospy.get_param("~loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)
    rospy.on_shutdown(cls.delete)

    while not rospy.is_shutdown():
        try:
            pass
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        loop_wait.sleep()
