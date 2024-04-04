#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import os
import math
import json

import roslib
import rospy
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    return dict(map(ascii_encode, pair) for pair in data.items())

if __name__ == "__main__":
    # ROS initializa
    rospy.init_node("env_marker_node")
    pub = rospy.Publisher("/hma_pointing_pkg/env_marker", MarkerArray, queue_size=1)
    furniture_json_file = rospy.get_param(rospy.get_name() + "/furniture_json", "furniture.json")
    arena_json_file     = rospy.get_param(rospy.get_name() + "/arena_json", "arena.json")

    # get json
    with open( furniture_json_file) as f:
        furniture_json = json.load(f, object_hook=ascii_encode_dict)

    with open( arena_json_file) as f:
        arena_json = json.load(f, object_hook=ascii_encode_dict)

    # print(json_data)

    while not rospy.is_shutdown():
        marker_array = MarkerArray()

        # arena
        for plane_data in arena_json:
            frame = plane_data["frame"]
            arena_name = plane_data["name"]
            for idx, plane in enumerate(plane_data["plane"]):

                plane_array = np.array(plane)
                center_pos = np.mean(plane_array, axis=0)

                # text marker
                marker_text = Marker()
                marker_text.header.frame_id = frame
                marker_text.header.stamp = rospy.Time.now()
                marker_text.ns = arena_name + "_text"
                marker_text.type = Marker.TEXT_VIEW_FACING
                marker_text.lifetime = rospy.Duration(1)
                marker_text.action = Marker.ADD
                marker_text.color.r = 1.0
                marker_text.color.g = 1.0
                marker_text.color.b = 1.0
                marker_text.color.a = 1.0
                marker_text.pose.orientation.w = 1.0
                marker_text.scale.z = 0.15
                marker_text.pose.position.x = center_pos[0]
                marker_text.pose.position.y = center_pos[1]
                marker_text.pose.position.z = 0.2
                marker_text.text = arena_name
                marker_array.markers.append(marker_text)

                marker_arena = Marker()
                marker_arena.header.frame_id = frame
                marker_arena.header.stamp = rospy.Time.now()
                marker_arena.ns = arena_name + "_area"
                marker_arena.type = Marker.CUBE
                marker_arena.action = Marker.ADD
                marker_arena.lifetime = rospy.Duration(1)
                marker_arena.color.r = 1.0
                marker_arena.color.g = 0
                marker_arena.color.b = 0
                marker_arena.color.a = 0.5

                # plane_array = np.array(plane)
                # center_pos = np.mean(plane_array, axis=0)

                marker_arena.pose.position.x = center_pos[0]
                marker_arena.pose.position.y = center_pos[1]
                marker_arena.pose.position.z = 0.01

                marker_arena.scale.x = abs(plane_array[:, 0].max() - plane_array[:, 0].min())
                marker_arena.scale.y = abs(plane_array[:, 1].max() - plane_array[:, 1].min())
                marker_arena.scale.z = 0.01

                marker_array.markers.append(marker_arena)


        # furniture
        for plane_data in furniture_json:
            furniture_name = plane_data["name"]
            frame = plane_data["frame"]

            # text marker
            marker_text = Marker()
            marker_text.header.frame_id = frame
            marker_text.header.stamp = rospy.Time.now()
            marker_text.ns = furniture_name + "_text"
            marker_text.type = Marker.TEXT_VIEW_FACING
            marker_text.lifetime = rospy.Duration(1)
            marker_text.action = Marker.ADD
            marker_text.color.r = 1.0
            marker_text.color.g = 1.0
            marker_text.color.b = 1.0
            marker_text.color.a = 1.0
            marker_text.pose.orientation.w = 1.0
            marker_text.scale.z = 0.15
            marker_text.pose.position.x = plane_data["plane"][0][0][0]
            marker_text.pose.position.y = plane_data["plane"][0][0][1]
            marker_text.pose.position.z = plane_data["plane"][0][0][2]
            marker_text.text = furniture_name
            marker_array.markers.append(marker_text)

            for idx, plane in enumerate( plane_data["plane"]):
                marker_furniture = Marker()
                marker_furniture.header.frame_id = frame
                marker_furniture.ns = furniture_name + "_" + str(idx)
                marker_furniture.header.stamp = rospy.Time.now()
                marker_furniture.type = Marker.CUBE
                marker_furniture.action = Marker.ADD
                marker_text.lifetime = rospy.Duration(1)
                marker_furniture.color.r = 0
                marker_furniture.color.g = 1.0
                marker_furniture.color.b = 0
                marker_furniture.color.a = 0.5

                plane_array = np.array(plane)
                center_pos = np.mean(plane_array, axis=0)

                marker_furniture.pose.position.x = center_pos[0]
                marker_furniture.pose.position.y = center_pos[1]
                marker_furniture.pose.position.z = center_pos[2]

                # x-dim
                if plane_array[:, 0].max() == plane_array[:, 0].min():
                    marker_furniture.scale.x = 0.01
                else:
                    marker_furniture.scale.x = abs(plane_array[:, 0].max() - plane_array[:, 0].min())

                # y-dim
                if plane_array[:, 1].max() == plane_array[:, 1].min():
                    marker_furniture.scale.y = 0.01
                else:
                    marker_furniture.scale.y = abs(plane_array[:, 1].max() - plane_array[:, 1].min())

                # z-dim
                if plane_array[:, 2].max() == plane_array[:, 2].min():
                    marker_furniture.scale.z = 0.01
                else:
                    marker_furniture.scale.z = abs(plane_array[:, 2].max() - plane_array[:, 2].min())


                # for point in plane:
                #     marker_furniture.points.append(Point(point[0], point[1], point[2]))
                marker_array.markers.append(marker_furniture)

        # publish marker
        # print(marker_array)
        pub.publish(marker_array)
        rospy.sleep(1.0)
