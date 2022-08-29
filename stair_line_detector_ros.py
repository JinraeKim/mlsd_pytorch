#!/usr/bin/env python3
from __future__ import print_function
import numpy as np

import sys
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.point_cloud2 import create_cloud, create_cloud_xyz32, read_points
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from stair_line_detector import StairLineDetector, LinearRegressor


class StairLineDetectorROS:
    def __init__(self, visualisation=True):
        self.stair_line_detector = StairLineDetector()
        self.bridge = CvBridge()
        self.visualisation = visualisation

        self.color_pub = rospy.Publisher("/camera/color/image_raw_with_lines", Image)
        self.pcd_pub = rospy.Publisher("/camera/depth/color/points_lines", PointCloud2)

        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.pcd_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)

        self.sync_sub = message_filters.TimeSynchronizer([self.color_sub, self.pcd_sub], 10)
        self.sync_sub.registerCallback(self.callback)

    def callback(self, color_data, pcd):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # (height, width, channels) = cv_color.shape

        # lines = self.stair_line_detector.detect_lines(cv_color)
        # cv_color = self.stair_line_detector.visualise(cv_color, lines, color=(255, 0, 0))

        _lines = self.stair_line_detector._detect_lines(cv_color)
        lines = self.stair_line_detector._filter_outlier_out(_lines)
        # lines = [lines[-1]]  # TODO: REMOVE IT
        line_iterators = [
            self.create_line_iterator(line[0:2], line[2:4], cv_color)
            for line in lines
        ]
        _points_iterators = [
            read_points(
                pcd,
                field_names=["x", "y", "z"],
                uvs=line_iterator,
                skip_nans=True,
            )
            for line_iterator in line_iterators
        ]
        points_iterators = [
            [p for p in pi] for pi in _points_iterators
        ]  # this is necessary cuz points_iterators is removed after any iteration
        yaw_estimated = self.estimate_yaw(points_iterators)
        print("estimated yaw", np.rad2deg(yaw_estimated))
        if self.visualisation:
            # lines
            cv_color = self.stair_line_detector.visualise(cv_color, _lines, color=(0, 0, 255))  # red
            cv_color = self.stair_line_detector.visualise(cv_color, lines, color=(255, 0, 0))  # blue
            try:
                self.color_pub.publish(self.bridge.cv2_to_imgmsg(cv_color, "bgr8"))
            except CvBridgeError as e:
                print(e)

            # pointclouds
            points_all = []
            for pi in points_iterators:
                points_all = points_all + list(pi)
            print("points", len(points_all))
            print("lines", len(line_iterators))
            line_points_number = 0
            for li in line_iterators:
                line_points_number += len(li)
            print("lines_points", line_points_number)
            points_all = list(set(points_all))
            pcd_all = create_cloud_xyz32(pcd.header, points_all)
            # pcd_all = create_cloud(pcd.header, pcd.fields, points_all)
            self.pcd_pub.publish(pcd_all)
        # import pdb; pdb.set_trace()

    def _estimate_yaw(self, points, threshold):
        """
        Inputs:
            points: points in 3D spaces (frame: `camera_color_optical_frame`)
        Outputs:
            yaw: yaw angle in [-pi/2, pi/2] (would be w.r.t. a fixed frame)
        """
        points_x = np.array([p[0] for p in points]).reshape(-1, 1)
        points_z = np.array([p[2] for p in points])
        regressor = LinearRegressor()
        regressor.fit(points_x, points_z)
        ssr = np.sum(np.square(points_z - regressor.predict(points_x)))  # sum of squared residuals
        success = (ssr < threshold) and regressor.success
        bias, slope = regressor.params
        yaw = np.arctan(slope)
        return yaw, success

    def estimate_yaw(self, points_iterators, threshold=1e-2):
        yaw_and_successes = [self._estimate_yaw(points, threshold) for points in points_iterators]
        yaws = [yas[0] for yas in yaw_and_successes if yas[1]]
        print("selected/total yaws", ": ", len(yaws), "/", len(yaw_and_successes))
        return np.mean(yaws)

    def create_line_iterator(self, P1, P2, img,):
        """
        Produces an array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x, y)
            -P2: a numpy array that consists of the coordinate of the second point (x, y)
        Returns:
            -it: a numpy array that consists of the coordinates of each pixel in the radii (shape: [numPixels, 2], row = [x, y])
        Notes:
            Modified from [1]
        Refs:
            [1] https://stackoverflow.com/a/32857432
        """
        # define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        P1X = np.array(int(P1X))
        P1Y = np.array(int(P1Y))
        P2X = np.array(int(P2X))
        P2Y = np.array(int(P2Y))

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2), dtype=np.int)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                # itbuffer[:, 1] = np.arange(P1Y, P1Y-dYa-1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                # itbuffer[:, 1] = np.arange(P1Y, P1Y+dYa+1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                # itbuffer[:, 0] = np.arange(P1X, P1X-dXa-1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                # itbuffer[:, 0] = np.arange(P1X, P1X+dXa+1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                    # itbuffer[:, 1] = np.arange(P1Y-1, P1Y-dYa-1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                    # itbuffer[:, 1] = np.arange(P1Y+1, P1Y+dYa+1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
                # itbuffer[:, 0] = (slope*(itbuffer[:, 1]-P1Y+1)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                    # itbuffer[:, 0] = np.arange(P1X, P1X-dXa-1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                    # itbuffer[:, 0] = np.arange(P1X, P1X+dXa+1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y
                # itbuffer[:, 1] = (slope*(itbuffer[:, 0]-P1X+1)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        # itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
        line_iterator = [list(itbuffer[i, :]) for i in range(itbuffer.shape[0])]
        return line_iterator


def main(args):
    sdros = StairLineDetectorROS()
    rospy.init_node('image_depth_synchronise', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
