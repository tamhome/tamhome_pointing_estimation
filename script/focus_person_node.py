#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import json
import os
import math
import threading

import roslib
import rospy
import tf
import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Quaternion
from hma_ailia_msgs.msg import Ax3DPoseArray


sys.path.append(roslib.packages.get_pkg_dir("hma_lib_pkg") + "/script/src/lib")
from libtf import *
from libutil import *

#==================================================

# グローバル

#==================================================
GP_LOOP_RATE = 30.0
GP_POSE_THRESHOLD = 0.3
GP_TIME_FOR_START  = 1.0
GP_ENV_JSON_FOLDER = roslib.packages.get_pkg_dir("hma_pointing_pkg") + "/json/"


#==================================================

## @class CalculatePontingPositionNode

#==================================================
class FocusPersonNode(object):
    def __init__(self):
        self._lock = threading.Lock()

        self._init_ros_time = rospy.Time.now()
        self._update_ros_time = {}
        self._prev_ros_time = self._init_ros_time
        self._prv_omni_degree = 0
        self._prv_head_degree = 0
        self._run_enable = None
        self._mode = None
        self._rate = rospy.Rate(GP_LOOP_RATE)

        self._libutil = LibUtil()
        self._libtf = LibTF()
        self._robot_descriptor = self._libutil.getRobotDescriptor()

        #==================================================

        # ROSインタフェース

        #==================================================
        self._p_pose = rospy.get_param(
            rospy.get_name() + "/3d_pose",
            "/3d_pose"
        )

        self._p_omni_joint_state = "/hsrb/omni_base_controller/state"
        self._p_head_joint_state = "/hsrb/head_trajectory_controller/state"

        self._p_arena_json = rospy.get_param(
            rospy.get_name() + "/arena_json",
            "arena.json"
        )

        self._srv_run_enable = rospy.Service(
            rospy.get_name() + "/run_enable",
            SetBool,
            self.srvfSwitchRunEnable
        )

        self._pub_omni_base_controller = rospy.Publisher(
            "/hsrb/omni_base_controller/command",
            JointTrajectory,
            queue_size=1
        )

        self._pub_omni_base_controller = rospy.Publisher(
            "/hsrb/omni_base_controller/command",
            JointTrajectory,
            queue_size=1
        )

        self._pub_head_controller = rospy.Publisher(
            "/hsrb/head_trajectory_controller/command",
            JointTrajectory,
            queue_size=1
        )

        self._sub_smi_3d_pose = message_filters.Subscriber(
            self._p_pose,
            Ax3DPoseArray
        )

        self._sub_smi_omni_joint_state = message_filters.Subscriber(
            self._p_omni_joint_state,
            JointTrajectoryControllerState
        )

        self._sub_smi_head_joint_state = message_filters.Subscriber(
            self._p_head_joint_state,
            JointTrajectoryControllerState
        )

        interface = [self._sub_smi_3d_pose, self._sub_smi_omni_joint_state, self._sub_smi_head_joint_state]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            interface,
            10,
            0.1,
            allow_headerless=True
        )
        self._sync.registerCallback(
            self.subf3DPoseJointState
        )
        
        # run_enable
        self._run_enable = rospy.get_param(
            rospy.get_name() + "/start_node",
            False
        )

        # load arena.json
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


    #==================================================

    ## @fn srvfSwitchRunEnable
    ## @brief run_enable service
    ## @param
    ## @return

    #==================================================
    def srvfSwitchRunEnable(
        self,
        req
    ):
        res = SetBoolResponse()
        if req.data:
            self._run_enable = True
            res.message = "forcus person enabled"
        else:
            self._run_enable = False
            res.message = "forcus person disabled"
        res.success = True
        return res


    #==================================================

    ## @fn subf3DPoseOmniJointState
    ## @brief
    ## @param
    ## @return

    #==================================================
    def subf3DPoseJointState(
        self,
        smi_3d_pose_array,
        smi_omni_joint_state,
        smi_head_joint_state
    ):
        self._lock.acquire()
        self._smi_3d_pose_array = smi_3d_pose_array
        self._person_frame = smi_3d_pose_array.header.frame_id
        self._smi_omni_joint_state = smi_omni_joint_state
        self._smi_head_joint_state = smi_head_joint_state
        self._lock.release()

        fn = sys._getframe().f_code.co_name
        self._update_ros_time[fn] = rospy.Time.now()

        return


    def ascii_encode_dict(
        self,
        data
    ):
        ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
        return dict(map(ascii_encode, pair) for pair in data.items())

    #==================================================

    ## @fn Personの座標（base_footprint座標系）を取得する関数
    ## @brief
    ## @param
    ## @return

    #==================================================
    def getPersonPositions(
        self,
        conv_frame,
        person_frame,
        person_x,
        person_y,
        person_z,
    ):
        perspn_pose_from_base_footprint = self._libtf.getPoseWithOffset(
            conv_frame,
            person_frame,
            "target_person",
            Pose(
                Point(person_x, person_y, person_z),
                Quaternion(0,0,0,1.0)
            )
        )

        x = perspn_pose_from_base_footprint.position.x
        y = perspn_pose_from_base_footprint.position.y
        z = perspn_pose_from_base_footprint.position.z

        (roll, pitch, yaw) = euler_from_quaternion(
            [
                perspn_pose_from_base_footprint.orientation.x,
                perspn_pose_from_base_footprint.orientation.y,
                perspn_pose_from_base_footprint.orientation.z,
                perspn_pose_from_base_footprint.orientation.w
            ]
        )


        return x, y, z, yaw

    #==================================================

    ## @fn proc
    ## @brie main function
    ## @param
    ## @return

    #==================================================
    def proc(
        self
    ):
        rospy.loginfo("Finish Initialization. If you want to start node, Plese call rosservice {}/run_enable".format(os.path.basename(__file__).split(".")[0]))
        while not rospy.is_shutdown():
            if not self._run_enable:
                rospy.sleep(0.3)
                continue

            # initialize shoulder_pose
            angle = 0
            shoulder_pose = []

            # prepare message
            target_omni_joint  = JointTrajectory()
            target_omni_joint_points = JointTrajectoryPoint()
            target_head_joint  = JointTrajectory()
            target_head_joint_points = JointTrajectoryPoint()

            #==================================================

            # 最新の3DPoseを取得

            #==================================================
            current_ros_time = rospy.Time.now()
            while not rospy.is_shutdown():
                if "subf3DPoseJointState" not in self._update_ros_time.keys():
                    continue
                if self._update_ros_time["subf3DPoseJointState"] > current_ros_time:
                    break
            self._lock.acquire()

            if len(self._smi_3d_pose_array.people) == 0:
                self._mode = "search"
            else:
                shoulder_pose = []
                if self._smi_3d_pose_array.people[0].shoulder_center.score > GP_POSE_THRESHOLD:
                    shoulder_pose.append(self._smi_3d_pose_array.people[0].shoulder_center.point)
                if self._smi_3d_pose_array.people[0].shoulder_left.score > GP_POSE_THRESHOLD:
                    shoulder_pose.append(self._smi_3d_pose_array.people[0].shoulder_left.point)
                if self._smi_3d_pose_array.people[0].shoulder_right.score > GP_POSE_THRESHOLD:
                    shoulder_pose.append(self._smi_3d_pose_array.people[0].shoulder_right.point)

                if len(shoulder_pose) == 0:
                    # check wheather to up or down head_tild
                    if self._smi_3d_pose_array.people[0].nose > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].eye_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].eye_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].ear_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].ear_left > GP_POSE_THRESHOLD:
                        self._mode = "down_head_tilt"
                        
                    elif self._smi_3d_pose_array.people[0].elbow_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].elbow_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].wrist_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].wirst_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].hip_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].hip_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].knee_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].knee_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].ankle_right > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].ankle_left > GP_POSE_THRESHOLD or \
                    self._smi_3d_pose_array.people[0].body_center > GP_POSE_THRESHOLD:
                        self._mode = "up_head_tilt"
                    
                    else:
                        self._mode = "search"

                else:
                    self._mode = "focus"
                    # check person in arena
                    x, y, z, _ = self.getPersonPositions(
                            self._robot_descriptor["FRAME_MAP"],
                            self._person_frame,
                            shoulder_pose[0].x,
                            shoulder_pose[0].y,
                            shoulder_pose[0].z
                        )
                    for plane_data in self._arena_json:
                        for plane in plane_data["plane"]:
                            plane_array = np.array(plane)
                            if x > plane_array[:,0].max() or x < plane_array[:,0].min() or \
                            y > plane_array[:,1].max() or y < plane_array[:,1].min():
                                self._mode = "search"


            # prepare publish omni data
            omni_desired_positions = self._smi_omni_joint_state.desired.positions
            target_omni_joint.joint_names = self._smi_omni_joint_state.joint_names
            target_omni_joint_points.positions  = [omni_desired_positions[0], omni_desired_positions[1], omni_desired_positions[2]]
            target_omni_joint_points.velocities = [0.0, 0.0, 0.0]
            target_omni_joint_points.time_from_start = rospy.Time(GP_TIME_FOR_START)

            # prepare publish head data
            head_desired_positions = self._smi_head_joint_state.desired.positions
            target_head_joint.joint_names = self._smi_head_joint_state.joint_names
            target_head_joint_points.positions  = [head_desired_positions[0], head_desired_positions[1]]
            target_head_joint_points.velocities = [0.0, 0.0]
            target_head_joint_points.time_from_start = rospy.Time(GP_TIME_FOR_START)

            # missing or seach person
            if self._mode == "search":
                rospy.loginfo("MISIING PERSON. Past angle is {}".format(self._prv_omni_degree))
                if self._prv_omni_degree >= 0:
                    target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2] + np.deg2rad(20.0).item()
                    target_omni_joint_points.velocities[2] = 0.40
                else:
                    target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2] + np.deg2rad(-20.0).item()
                    target_omni_joint_points.velocities[2] = -0.40

                target_omni_joint.points = [target_omni_joint_points]
                self._pub_omni_base_controller.publish(target_omni_joint)

                # Head tilt
                target_head_joint_points.positions[0] = 0
                target_head_joint.points = [target_head_joint_points]
                self._prv_head_degree = 0
                self._pub_head_controller.publish(target_head_joint)

            # up_head
            elif self._mode == "up_head_tilt":
                head_angle = np.deg2rad(-25.0)
                target_head_joint_points.positions[0] = -head_angle
                target_head_joint.points = [target_head_joint_points]
                self._pub_head_controller.publish(target_head_joint)
                self._prv_head_degree = head_angle
                
            # donw_head
            elif self._mode == "down_head_tilt":
                head_angle = np.deg2rad(25.0)
                target_head_joint_points.positions[0] = -head_angle
                target_head_joint.points = [target_head_joint_points]
                self._pub_head_controller.publish(target_head_joint)
                self._prv_head_degree = head_angle
                


            # focus person
            elif self._mode == "focus":
                # get person pose from base_link
                x, y, _, _ = self.getPersonPositions(
                    self._robot_descriptor["FRAME_BASE_FOOTPRINT"],
                    self._person_frame,
                    shoulder_pose[0].x,
                    shoulder_pose[0].y,
                    shoulder_pose[0].z
                )
                angle = math.atan2(y, x)
                # rospy.loginfo("body angle:{}".format(angle))
                self._prv_omni_degree = angle

                if abs(angle) > np.deg2rad(10.0).item():
                    if angle < 0:
                        vel_t = -0.30
                    else:
                        vel_t = 0.30
                    target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2] + angle
                    target_omni_joint.points = [target_omni_joint_points]
                    target_omni_joint_points.velocities[2] = vel_t
                    self._pub_omni_base_controller.publish(target_omni_joint)
                else:
                    target_omni_joint_points.positions[2] = target_omni_joint_points.positions[2]
                    target_omni_joint.points = [target_omni_joint_points]
                    target_omni_joint_points.velocities[2] = 0
                    self._pub_omni_base_controller.publish(target_omni_joint)

                # Head tilt
                head_angle = math.atan2(shoulder_pose[0].y, shoulder_pose[0].z)
                # rospy.loginfo("prv_angle{} head_angle{}".format(self._prv_head_degree, head_angle))
                if abs(self._prv_head_degree - head_angle) > np.deg2rad(10) and \
                abs(head_angle) < np.deg2rad(25.0):
                    target_head_joint_points.positions[0] = -head_angle
                    target_head_joint.points = [target_head_joint_points]
                    self._prv_head_degree = head_angle
                    self._pub_head_controller.publish(target_head_joint)

            self._lock.release()
            self._prev_ros_time = current_ros_time
            self._rate.sleep()


if __name__ == "__main__":
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    p_loop_rate = rospy.get_param(
        rospy.get_name() + "/loop_rate",
        GP_LOOP_RATE
    )
    loop_wait = rospy.Rate(p_loop_rate)

    ps_node = FocusPersonNode()
    ps_node.proc()

    rospy.on_shutdown(ps_node.delete)

    while not rospy.is_shutdown():
        try:
            pass
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        loop_wait.sleep()
