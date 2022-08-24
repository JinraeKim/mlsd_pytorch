#!/usr/bin/env python3
from __future__ import print_function
import numpy as np

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from stair_line_detector import StairLineDetector


class StairLineDetectorROS:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.image_depth_sync = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.image_depth_sync.registerCallback(self.callback)
        self.stair_line_detector = StairLineDetector()

    def callback(self, image_data, depth_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # lines = self.stair_line_detector.pred_lines(cv_image)
        # cv_image = self.stair_line_detector.visualise(cv_image, lines, color=(256, 0, 0))

        _lines = self.stair_line_detector._pred_lines(cv_image)
        lines = self.stair_line_detector._filter_outlier_out(_lines)
        cv_image = self.stair_line_detector.visualise(cv_image, _lines)
        cv_image = self.stair_line_detector.visualise(cv_image, lines, color=(256, 0, 0))

        # (rows, cols, channels) = cv_image.shape
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    ids = StairLineDetectorROS()
    rospy.init_node('image_depth_synchronise', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
