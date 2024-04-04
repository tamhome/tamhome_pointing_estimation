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
import threading
import random

from tamlib.utils import Logger

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse


class PointingEstimation(Logger):
    def __init__(self):
        Logger.__init__(self)

    def run(self):
        """指差し認識の動作関数
        """