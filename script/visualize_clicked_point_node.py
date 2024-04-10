#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped


class visualizeClickePoint(object):
    def __init__(self):
        self._sub = rospy.Subscriber("/clicked_point", PointStamped, self.callback)
        self._pub = rospy.Publisher(rospy.get_name() + "/clicked_point",  Marker, queue_size=1)

    def callback(self, msg):
        marker_text = Marker()
        marker_text.header.frame_id = msg.header.frame_id
        marker_text.header.stamp = rospy.Time.now()
        marker_text.ns = "clicked_point"
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.lifetime = rospy.Duration(0)
        marker_text.action = Marker.ADD
        marker_text.color.r = 1.0
        marker_text.color.g = 1.0
        marker_text.color.b = 1.0
        marker_text.color.a = 1.0
        marker_text.pose.orientation.w = 1.0
        marker_text.scale.z = 0.25
        marker_text.pose.position.x = msg.point.x
        marker_text.pose.position.y = msg.point.y
        marker_text.pose.position.z = msg.point.z

        marker_text.text = "({}, {})".format(str(msg.point.x), str(msg.point.y))
        self._pub.publish(marker_text)


if __name__ == "__main__":
    rospy.init_node("visualize_clicked_point_node")
    vcp = visualizeClickePoint()
    rospy.spin()
