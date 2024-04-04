#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#TODO: ActionServerに実装しなおす
#TODO: arena内外判定の実装

from __future__ import absolute_import
from audioop import cross
import sys
import time
import json
import os
import math
import threading
import random

import roslib
import rospy
import tf
import numpy as np

from hma_ailia_msgs.msg import Ax3DPoseArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse

sys.path.append(roslib.packages.get_pkg_dir("hma_lib_pkg") + "/script/src/lib")
from libtf import *
from libutil import *
#==================================================

# グローバル

#==================================================
GP_LOOP_RATE = 30.0
GP_POSE_THRESHOLD = 0.3
GP_POINTING_THRESHOLD = 0.15
GP_ELBOW_DISANCE_THRESHOLD = 0.15



#==================================================

## @class NumpyRingBuffer

#==================================================
class NumpyRingBuffer(object):
    def __init__(self, data_shape=np.array([0,0,0]), buffer_size=5):
        self._data_shape = data_shape
        self._buffer_size = buffer_size
        self._data = [self._data_shape for i in range(self._buffer_size)]
        self._index = 0
        self._is_full = False

    def append(self, data):
        if not self._data_shape.shape ==  data.shape:
            print("Invalid data shape")
            return False
        self._data[self._index] = data
        self._index += 1
        if self._index >= self._buffer_size:
            self._index = 0
            self._is_full = True
        return True

    def get_data(self):
        return self._data

    def is_full(self):
        return self._is_full

    def data_initialize(self):
        self._data = [self._data_shape for i in range(self._buffer_size)]
        self._index = 0
        self._is_full = False

#==================================================

## @class CalculatePontingPositionNode

#==================================================

class CalculatePontingPositionNode(object):
    #==================================================

    ## @fn __init__
    ## @brief コンストラクタ
    ## @param
    ## @return

    #==================================================
    def __init__(self) :
        self._lock = threading.Lock()

        self._init_ros_time = rospy.Time.now()
        self._update_ros_time = {}
        self._prev_ros_time = self._init_ros_time

        self._smi_3d_pose_array = Ax3DPoseArray()

        self.pose_threhold = GP_POSE_THRESHOLD
        self.pointing_threshold = GP_POINTING_THRESHOLD
        self.elbow_distance_threshold = GP_ELBOW_DISANCE_THRESHOLD

        self._tf_listener = tf.TransformListener()
        self._left_arm_position_array = NumpyRingBuffer(data_shape=np.array([[0,0,0],[0,0,0],[0,0,0]]),buffer_size=10)
        self._right_arm_position_array = NumpyRingBuffer(data_shape=np.array([[0,0,0],[0,0,0],[0,0,0]]),buffer_size=10)
        self._point_position_array = NumpyRingBuffer(data_shape=np.array([0,0,0]), buffer_size=30)
        self._current_pointed_furniture = None
        self._prv_pointed_furniture     = None
        self._furniture_json    = None
        self._arena_json    = None
        self._cam_model   = None
        self._base_frame  = None
        self._run_enable  = None

        #==================================================

        # ROSインタフェース

        #==================================================
        self._p_pose = rospy.get_param(
            rospy.get_name() + "/3d_pose",
            "/3d_pose"
        )

        self._p_furniture_json = rospy.get_param(
            rospy.get_name() + "/furniture_json",
            "furniture.json"
        )

        self._p_arena_json = rospy.get_param(
            rospy.get_name() + "/arena_json",
            "arena.json"
        )

        self._p_fix_position = rospy.get_param(
            rospy.get_name() + "/fix_position",
            True
        )

        self._pub_pointing_line = rospy.Publisher(
            rospy.get_name() + "/pointing_line",
            Marker,
            queue_size = 1
        )

        self._pub_pointing_position= rospy.Publisher(
            rospy.get_name() + "/pointing_position",
            Marker,
            queue_size = 1
        )

        self._sub_3d_pose_array = rospy.Subscriber(
            self._p_pose,
            Ax3DPoseArray,
            self.sub3DPoseArray,
            queue_size=1
        )

        self._srv_run_enable = rospy.Service(
            rospy.get_name() + "/run_enable",
            SetBool,
            self.srvfSwitchRunEnable
        )

        # run_enable
        self._run_enable = rospy.get_param(
            rospy.get_name() + "/start_node",
            False
        )

        # load json file
        with open(self._p_furniture_json ) as f:
            self._furniture_json = json.load(f, object_hook=self.ascii_encode_dict)

        with open(self._p_arena_json ) as f:
            self._arena_json = json.load(f, object_hook=self.ascii_encode_dict)


    #==================================================

    ## @fn delete
    ## @brief デストラクタ
    ## @param
    ## @return

    #==================================================
    def delete(
        self
    ):
        #==================================================

        # ファイナライズ

        #==================================================


        return


    def sub3DPoseArray(
        self,
        msg,
    ):
        self._lock.acquire()
        self._smi_3d_pose_array = msg
        self._base_frame = msg.header.frame_id
        self._lock.release()
        fn = sys._getframe().f_code.co_name
        self._update_ros_time[fn] = rospy.Time.now()
        return


    def srvfSwitchRunEnable(
        self,
        req
    ):
        res = SetBoolResponse()
        if req.data:
            self._run_enable = True
            res.message = "calculate_position enabled"
        else:
            self._run_enable = False
            res.message = "calculate_postion disabled"
        res.success = True
        return res


    def ascii_encode_dict(
        self,
        data
    ):
        ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
        return dict(map(ascii_encode, pair) for pair in data.items())


    def displayPointingLine(
        self,
        np_Point1,
        np_Point2,
        target_frame
    ):
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
        marker.points.append(Point(np_Point1[0], np_Point1[1], np_Point1[2]))
        marker.points.append(Point(np_Point2[0], np_Point2[1], np_Point2[2]))
        self._pub_pointing_line.publish(marker)


    def displayPointingPosition(
        self,
        crosss_point,
        target_frame,
        lifetime = 0.1
    ):
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


    def debugDisplayKeyPoint(
        self,
        np_Point1
    ):
        marker = Marker()
        marker.header.frame_id = self._base_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "keypoint"
        marker.lifetime = rospy.Duration(1)
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 1
        marker.color.r = 1.0
        marker.color.b = 0.0
        marker.color.g = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        print(np_Point1[0], np_Point1[1], np_Point1[2])
        marker.points.append(Point(np_Point1[0], np_Point1[1], np_Point1[2]))
        print(marker)
        self._pub_pointing_line.publish(marker)


    def checkStraightArm(
        self,
        target_arm_position,
    ):
        center_position = ( target_arm_position[0] + target_arm_position[2] ) / 2.0
        disntance_elbow_to_center = np.linalg.norm( target_arm_position[1] - center_position )
        # print(disntance_elbow_to_center)
        if disntance_elbow_to_center > self.elbow_distance_threshold:
            return False
        else:
            return True


    def transformPoint(
        self,
        target_frame,
        base_frame,
        point
    ):
        while not rospy.is_shutdown():
            try:
                (trans,rot) = self._tf_listener.lookupTransform(target_frame, base_frame,rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        rot_matrix = tf.transformations.quaternion_matrix(rot)
        trans_matrix = tf.transformations.translation_matrix(trans)
        plane_vec = np.array(
            [
            point[0],
            point[1],
            point[2],
            1.0
            ]
        )
        transform_matrix = np.dot(trans_matrix, rot_matrix)
        transed_plane_vec = np.dot(transform_matrix, plane_vec)
        return np.array([transed_plane_vec[0], transed_plane_vec[1], transed_plane_vec[2]])


    def checkCrossPlane(
        self,
        target_wrist_point,
        target_direction_vector,
        np_planes
    ):
        for np_plane in np_planes:
            plane_center_point = np.mean(np_plane, axis=0)

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
                continue

            # calculate point
            # param_t = ( np_plane[0][axis] - target_wrist_point[axis] ) / target_direction_vector[axis]
            # if param_t <= 0.0:
            #     continue
            param_t = ( np_plane[0][axis] - target_wrist_point[axis] ) / target_direction_vector[axis]
            if param_t <= 0.0 or param_t >=3.5:
                continue
            cross_point = target_wrist_point + param_t * target_direction_vector

            # chech inside or outside
            if axis==0 and \
            abs( cross_point[1] - plane_center_point[1]) <= abs(np_plane[:, 1].max() - np_plane[:, 1].min()) / 2.0 and \
            abs( cross_point[2] - plane_center_point[2]) <= abs(np_plane[:, 2].max() - np_plane[:, 2].min()) / 2.0:
                return cross_point
            elif axis==1 and \
            abs( cross_point[0] - plane_center_point[0]) <= abs(np_plane[:, 0].max() - np_plane[:, 0].min()) / 2.0 and \
            abs( cross_point[2] - plane_center_point[2]) <= abs(np_plane[:, 2].max() - np_plane[:, 2].min()) / 2.0:
                return cross_point
            elif axis==2 and \
            abs( cross_point[0] - plane_center_point[0]) <= abs(np_plane[:, 0].max() - np_plane[:, 0].min()) / 2.0 and \
            abs( cross_point[1] - plane_center_point[1]) <= abs(np_plane[:, 1].max() - np_plane[:, 1].min()) / 2.0:
                return cross_point
        return None



    def proc(
        self
    ):
        rospy.loginfo("Finish Initialization. If you want to start node, Plese call rosservice {}/run_enable".format(os.path.basename(__file__).split(".")[0]))
        while not rospy.is_shutdown():
            if not self._run_enable:
                self._current_pointed_furniture = self._prv_pointed_furniture = 0
                self._point_position_array.data_initialize()
                rospy.sleep(0.3)
                continue

            #==================================================

            # 最新の3DPoseを取得

            #==================================================
            current_ros_time = rospy.Time.now()
            while not rospy.is_shutdown():
                if "sub3DPoseArray" not in self._update_ros_time.keys():
                    continue
                if self._update_ros_time["sub3DPoseArray"] > current_ros_time:
                    break
            self._lock.acquire()

            # check person num
            if len(self._smi_3d_pose_array.people) == 0:
                rospy.logwarn("No person detected")
                self._lock.release()
                self._prev_ros_time  = current_ros_time
                continue
            elif len(self._smi_3d_pose_array.people) > 1:
                rospy.logwarn("More than one person detected")
                self._lock.release()
                self._prev_ros_time  = current_ros_time
                continue

            target_person = self._smi_3d_pose_array.people[0]
            left_arm_pose  = None
            right_arm_pose = None

            # check arm pose is available
            if target_person.shoulder_left.score > self.pose_threhold or \
            target_person.elbow_left.score > self.pose_threhold or \
            target_person.wrist_left.score > self.pose_threhold :
                left_arm_pose = np.array( [[target_person.shoulder_left.point.x, target_person.shoulder_left.point.y, target_person.shoulder_left.point.z],
                                           [target_person.elbow_left.point.x, target_person.elbow_left.point.y, target_person.elbow_left.point.z],
                                           [target_person.wrist_left.point.x, target_person.wrist_left.point.y, target_person.wrist_left.point.z]] )
                self._left_arm_position_array.append(left_arm_pose)
                # debug
                # print(left_arm_pose[2])
                # self.debugDisplayKeyPoint(left_arm_pose[2])

            if target_person.shoulder_right.score > self.pose_threhold or \
            target_person.elbow_right.score > self.pose_threhold or \
            target_person.wrist_right.score > self.pose_threhold :
                right_arm_pose = np.array( [[target_person.shoulder_right.point.x, target_person.shoulder_right.point.y, target_person.shoulder_right.point.z],
                                            [target_person.elbow_right.point.x, target_person.elbow_right.point.y, target_person.elbow_right.point.z],
                                            [target_person.wrist_right.point.x, target_person.wrist_right.point.y, target_person.wrist_right.point.z]] )
                self._right_arm_position_array.append(right_arm_pose)

            ##########################################
            # check which arm to use
            ##########################################
            ## If neither arm is available, skip
            if left_arm_pose is None and right_arm_pose is None:
                self._lock.release()
                rospy.logwarn("Please capture body")
                continue

            # ## one arm is available
            elif left_arm_pose is None:
                rospy.loginfo("Focus right arm")
                if self._right_arm_position_array.is_full:
                    target_positon_median  = np.median(self._right_arm_position_array.get_data(), axis=0)
                    target_wrist_point     = target_positon_median[2]
                    target_shoulder_point  = target_positon_median[0]
                else:
                    self._lock.release()
                    continue
            elif right_arm_pose is None:
                rospy.loginfo("Focus left arm")
                if self._left_arm_position_array.is_full:
                    target_positon_median  = np.median(self._left_arm_position_array.get_data(), axis=0)
                    target_wrist_point     = target_positon_median[2]
                    target_shoulder_point  = target_positon_median[0]
                else:
                    self._lock.release()
                    continue

            ## if both arm is available
            else:
                left_arm_direction_vector = left_arm_pose[2] - left_arm_pose[0]
                right_arm_direction_vector = right_arm_pose[2] - right_arm_pose[0]
                left_shoulder2elbow_direction_vector = left_arm_pose[1] - left_arm_pose[0]
                right_shoulder2elbow_direction_vector = right_arm_pose[1] - right_arm_pose[0]

                # print(left_arm_direction_vector)
                # print(right_arm_direction_vector)

                if abs( left_arm_direction_vector[0]) < self.pointing_threshold and \
                abs( left_arm_direction_vector[2]) < self.pointing_threshold and \
                abs( right_arm_direction_vector[0]) < self.pointing_threshold and \
                abs( right_arm_direction_vector[2]) < self.pointing_threshold:
                    self._lock.release()
                    # rospy.loginfo("Person dont pointing. skip")
                    continue
                # elif abs( left_arm_direction_vector[0]) > abs( right_arm_direction_vector[0]) and self.checkStraightArm(left_arm_pose):
                #     rospy.loginfo("Focus left arm")
                #     target_arm_pose = left_arm_pose
                #     target_arm_directtion_vector = left_arm_direction_vector
                # elif abs( right_arm_direction_vector[0]) > abs( left_arm_direction_vector[0]) and self.checkStraightArm(right_arm_pose):
                #     rospy.loginfo("Focus right arm")
                #     target_arm_pose = right_arm_pose
                #     target_arm_directtion_vector = right_arm_direction_vector

                elif abs( left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) > abs( right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) and self.checkStraightArm(left_arm_pose):
                    # rospy.loginfo("Focus left arm")
                    # target_arm_pose = left_arm_pose
                    # target_arm_directtion_vector = left_arm_direction_vector
                    if self._left_arm_position_array.is_full:
                        target_positon_median  = np.median(self._left_arm_position_array.get_data(), axis=0)
                        target_wrist_point     = target_positon_median[2]
                        target_shoulder_point  = target_positon_median[0]
                    else:
                        self._lock.release()
                        continue
                elif abs( right_arm_direction_vector[0]) + abs(right_arm_direction_vector[2]) > abs( left_arm_direction_vector[0]) + abs(left_arm_direction_vector[2]) and self.checkStraightArm(right_arm_pose):
                    # rospy.loginfo("Focus right arm")
                    # target_arm_pose = right_arm_pose
                    # target_arm_directtion_vector = right_arm_direction_vector
                    if self._right_arm_position_array.is_full:
                        target_positon_median  = np.median(self._right_arm_position_array.get_data(), axis=0)
                        target_wrist_point     = target_positon_median[2]
                        target_shoulder_point  = target_positon_median[0]
                    else:
                        self._lock.release()
                        continue

                else:
                    self._lock.release()
                    # rospy.loginfo("Cant detet pointing direction. skip")
                    continue

            # debug display marker at base_frame
            # self.displayPointingLine(target_wrist_point, target_wrist_point + (target_wrist_point - target_shoulder_point ) *3.0, target_frame=self._base_frame)

            # transform wrist_point to target_frame
            target_frame = self._furniture_json[0]["frame"]
            transed_target_wrist_point     = self.transformPoint( target_frame, self._base_frame, target_wrist_point)
            transed_target_shoulder_point  = self.transformPoint( target_frame, self._base_frame, target_shoulder_point)

            # debug display marker at target_frame
            self.displayPointingLine(transed_target_wrist_point , transed_target_wrist_point + ( transed_target_wrist_point - transed_target_shoulder_point) * 3.0 , target_frame=target_frame)

            # check cross plane
            for furniture in self._furniture_json:
                self._current_pointed_furniture = furniture["name"]
                self._current_height_th = furniture["height_th"]
                cross_point = self.checkCrossPlane(transed_target_wrist_point,  ( transed_target_wrist_point - transed_target_shoulder_point) , np.array( furniture["plane"]))
                if cross_point is not None:
                    self.displayPointingPosition(cross_point, target_frame)
                    self._point_position_array.append(cross_point)
                    break

            # check pointed furniture change
            if  self._current_pointed_furniture != self._prv_pointed_furniture and self._prv_pointed_furniture is not None:
                rospy.logwarn("Pointed Furniture Changed from {} to {}".format(self._prv_pointed_furniture, self._current_pointed_furniture))
                self._point_position_array.data_initialize()
                self._current_pointed_furniture = self._prv_pointed_furniture = None

            if self._point_position_array.is_full() and self._p_fix_position:
                pointed_position_median = np.median(self._point_position_array.get_data(), axis=0)
                pointed_position_std    = np.std(self._point_position_array.get_data(), axis=0)
                print(pointed_position_median)
                print(pointed_position_std)
                if np.all(pointed_position_std < 1.0):
                    rospy.loginfo("Pointed furniture: {}".format(self._current_pointed_furniture))
                    rospy.loginfo("Pointed Position: {}".format(pointed_position_median))
                    rospy.set_param(rospy.get_name() + "/furniture", self._current_pointed_furniture)
                    rospy.set_param(rospy.get_name() + "/position", pointed_position_median.tolist())
                    rospy.set_param(rospy.get_name() + "/height_th", self._current_height_th)
                    self.displayPointingPosition(pointed_position_median, target_frame, lifetime=0)
                    self._run_enable = False

            self._prv_pointed_furniture = self._current_pointed_furniture
            self._current_pointed_furniture = None

            #==================================================
            self._lock.release()
            self._prev_ros_time = current_ros_time


#==================================================

# メイン

#==================================================
if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    p_loop_rate = rospy.get_param(
        rospy.get_name() + "/loop_rate",
        GP_LOOP_RATE
    )
    loop_wait = rospy.Rate(p_loop_rate)

    ps_node = CalculatePontingPositionNode()
    ps_node.proc()

    rospy.on_shutdown(ps_node.delete)

    while not rospy.is_shutdown():
        try:
            pass
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        loop_wait.sleep()
