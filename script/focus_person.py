#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tf
import sys
import time
import json
import math
import rospy
import roslib
import threading
import numpy as np
import message_filters

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Quaternion, Pose, Point, Twist
# from hma_ailia_msgs.msg import Ax3DPoseArray
from tamlib.node_template import Node

from tamlib.tf import Transform, euler2quaternion, transform_pose
from tam_mmaction2.msg import Ax3DPoseWithLabelArray, Ax3DPoseWithLabel, AxKeyPoint


class FocusPersonNode(Node):
    def __init__(self):
        super().__init__(loglevel="INFO")

        # HSRに接続するためのインタフェースを確立
        self.tamtf = Transform()
        self.sigverse_flag = rospy.get_param("~is_sigverse", True)

        if self.sigverse_flag is False:
            # 実機用コード
            from hsrlib.utils import utils, description, joints, locations
            from hsrnavlib import LibHSRNavigation
            from hsrlib.hsrif import HSRInterfaces
            self.description = description.load_robot_description()
            self.hsrif = HSRInterfaces()
            self.hsrnav = LibHSRNavigation()
            # self.speech_recog = LibSpeechRecog()
        else:
            from sigverse_hsrlib import MoveJoints
            self.move_joint = MoveJoints()
            self.frame_map = "map"
            self.frame_baselink = "base_footprint"
            self.frame_rgbd = "head_rgbd_sensor_link"

        # ros interface
        self._p_pose = rospy.get_param("~3d_pose", "/3d_pose")
        self._p_arena_json = rospy.get_param("~arena_json", "arena.json")
        self.run_enable = rospy.get_param("~start_node", True)
        self.sholder_th = rospy.get_param("~sholder_th", 0.7)
        self.sholder_distance = rospy.get_param("sholder_distance", 1.0)

        self._p_omni_joint_state = "/hsrb/omni_base_controller/state"
        self._p_head_joint_state = "/hsrb/head_trajectory_controller/state"

        # self._srv_run_enable = rospy.Service("~run_enable", SetBool, self.srvfSwitchRunEnable)
        self._pub_omni_base_controller = rospy.Publisher("/hsrb/omni_base_controller/command", JointTrajectory, queue_size=1)
        self._pub_omni_base_controller = rospy.Publisher("/hsrb/omni_base_controller/command", JointTrajectory, queue_size=1)
        self._pub_head_controller = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=1)
        self._pub_base_rot = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
        self.initalize = True
        self.recovery_angular_velocity = 1.0

        # self._sub_smi_3d_pose = message_filters.Subscriber(self._p_pose, Ax3DPoseWithLabelArray)
        # self._sub_smi_omni_joint_state = message_filters.Subscriber(self._p_omni_joint_state, JointTrajectoryControllerState)
        # self._sub_smi_head_joint_state = message_filters.Subscriber(self._p_head_joint_state, JointTrajectoryControllerState)

        # interface = [self._sub_smi_3d_pose, self._sub_smi_omni_joint_state, self._sub_smi_head_joint_state]
        # self._sync = message_filters.ApproximateTimeSynchronizer(interface, 10, 0.1, allow_headerless=True)
        # self._sync.registerCallback(self.subf3DPoseJointState)

    def delete(self):
        """デストラクタ"""
        return

    def calculate_distance(self, keypoint_1: Point, keypoint_2: Point):
        """keypoint間の距離を計算する関数
        Args:
            keypoint_1(Point):
            keypoint_2(Point):
        Returns:
            float: 距離
        """
        x = keypoint_1.x - keypoint_2.x
        y = keypoint_1.y - keypoint_2.y
        z = keypoint_1.z - keypoint_2.z
        distance = math.sqrt(x**2 + y**2 + z**2)
        return distance

    def tracking_with_omni(self, human_info: Ax3DPoseWithLabel) -> None:
        """人の骨格情報をもとに人を追いかける
        Args:
            human_info(Ax3DPoseWithLabel): 追跡対象の人の骨格情報
        Retruns:
            None
        """
        # get person pose from base_link
        nose_point: Point = human_info.keypoints.nose.point
        self.logtrace(nose_point)

        # 実機
        if self.sigverse_flag is False:
            nose_pose_from_map: Pose = self.tamtf.get_pose_with_offset(
                target_frame=self.description.frame.map,
                source_frame=self.description.frame.rgbd,
                offset=Pose(nose_point, Quaternion(0, 0, 0, 1)),
            )

            nose_pose_from_baselink: Pose = self.tamtf.get_pose_with_offset(
                target_frame=self.description.frame.baselink,
                source_frame=self.description.frame.rgbd,
                offset=Pose(nose_point, Quaternion(0, 0, 0, 1)),
            )
        # sigverse
        else:
            nose_pose_from_map: Pose = self.tamtf.get_pose_with_offset(
                target_frame=self.frame_map,
                source_frame=self.frame_rgbd,
                offset=Pose(nose_point, Quaternion(0, 0, 0, 1)),
            )

            nose_pose_from_baselink: Pose = self.tamtf.get_pose_with_offset(
                target_frame=self.frame_baselink,
                source_frame=self.frame_rgbd,
                offset=Pose(nose_point, Quaternion(0, 0, 0, 1)),
            )

        x = nose_pose_from_baselink.position.x
        y = nose_pose_from_baselink.position.y

        angle = math.atan2(y, x)
        self._prv_omni_degree = angle
        twist_msg = Twist()

        if abs(angle) > np.deg2rad(10.0).item():
            if angle < 0:
                self.logdebug("右向きに旋回")
                twist_msg.angular.z = -0.06
                self._pub_base_rot.publish(twist_msg)
                self.recovery_angular_velocity = -1.0
            else:
                self.logdebug("左向きに旋回")
                twist_msg.angular.z = 0.06
                self._pub_base_rot.publish(twist_msg)
                self.recovery_angular_velocity = 1.0

                # vel_t = 0.15
                # self.move_joint.move_base_joint(0, 0, vel_t, duration=0.5)
            # target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2] + angle
            # target_omni_joint.points = [target_omni_joint_points]
            # target_omni_joint_points.velocities[2] = vel_t
            # self._pub_omni_base_controller.publish(target_omni_joint)
        else:
            # self.move_joint.move_base_joint(0, 0, 0, duration=0.1)
            twist_msg.angular.z = 0.0
            self._pub_base_rot.publish(twist_msg)
            self.logdebug("旋回を停止")
            # target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2]
            # target_omni_joint.points = [target_omni_joint_points]
            # target_omni_joint_points.velocities[2] = 0
            # self._pub_omni_base_controller.publish(target_omni_joint)

        # # Head tilt
        # head_angle = math.atan2(shoulder_pose[0].y, shoulder_pose[0].z)
        # # rospy.loginfo("prv_angle{} head_angle{}".format(self._prv_head_degree, head_angle))
        # if abs(self._prv_head_degree - head_angle) > np.deg2rad(10) and \
        # abs(head_angle) < np.deg2rad(25.0):
        #     target_head_joint_points.positions[0] = -head_angle
        #     target_head_joint.points = [target_head_joint_points]
        #     self._prv_head_degree = head_angle
        #     self._pub_head_controller.publish(target_head_joint)

        return None

    def run(self):
        """実行関数
        """
        while not rospy.is_shutdown():
            if self.run_enable is False:
                self.initalize = True
                self.logtrace("please run rosservice call /action_recognition_server/run_enable \"data: false\"")
                rospy.sleep(0.5)
                continue

            if self.initalize:
                self.initalize = False
                self.loginfo("start focus person node")

            # メッセージを取得する
            msg: Ax3DPoseWithLabelArray = rospy.wait_for_message("/mmaction2/poses/with_label", Ax3DPoseWithLabelArray)
            self.logdebug(f"detect {len(msg.people)} person")

            # 人が写っていない場合，旋回して探すモードに移行
            if len(msg.people) == 0:
                twist_msg = Twist()
                twist_msg.angular.z = self.recovery_angular_velocity
                self._pub_base_rot.publish(twist_msg)
                continue

            # 鼻の座標をもとに，一番近い人を見つける
            nearest_person_id = None
            min_distance = np.inf
            for id, detect_person in enumerate(msg.people):
                if detect_person.keypoints.nose.point.z < min_distance:
                    # 信頼値が低い結果ははじく
                    left_shoulder = detect_person.keypoints.left_shoulder
                    right_shoulder = detect_person.keypoints.right_shoulder
                    shoulder_distance = self.calculate_distance(left_shoulder.point, right_shoulder.point)
                    if detect_person.keypoints.left_shoulder.score < self.sholder_th or detect_person.keypoints.right_shoulder.score < self.sholder_th or shoulder_distance > 1.0:
                        continue
                    min_distance = detect_person.keypoints.nose.point.z
                    nearest_person_id = id

            # 一番近い人の情報をもとにしたトラッキング
            if nearest_person_id is not None:
                nearest_person_info: Ax3DPoseWithLabel = msg.people[nearest_person_id]
                self.tracking_with_omni(nearest_person_info)
            else:
                self.logdebug("信頼値が低いため棄却")


if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    p_loop_rate = rospy.get_param("~loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)

    ps_node = FocusPersonNode()
    ps_node.run()

    rospy.on_shutdown(ps_node.delete)

    while not rospy.is_shutdown():
        try:
            pass
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        loop_wait.sleep()
