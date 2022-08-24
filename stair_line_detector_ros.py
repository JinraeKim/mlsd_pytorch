#!/usr/bin/env python3
from __future__ import print_function
import numpy as np

import sys
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from stair_line_detector import StairLineDetector


class StairLineDetectorROS:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_lines", Image)
        self.tmp_pub = rospy.Publisher("image_topic_tmp", Image)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.image_camera_info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.image_depth_sync = message_filters.TimeSynchronizer([self.image_sub, self.image_camera_info_sub, self.depth_sub], 10)
        self.image_depth_sync.registerCallback(self.callback)
        self.stair_line_detector = StairLineDetector()

    def callback(self, image_data, image_camera_info, depth_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
        except CvBridgeError as e:
            print(e)
        # lines = self.stair_line_detector.detect_lines(cv_image)
        # cv_image = self.stair_line_detector.visualise(cv_image, lines, color=(255, 0, 0))

        K = np.array(image_camera_info.K).reshape(3, 3)

        _lines = self.stair_line_detector._detect_lines(cv_image)
        lines = self.stair_line_detector._filter_outlier_out(_lines)
        line_iterator = self.create_line_iterator(lines[0][0:2], lines[0][2:4], cv_depth)
        cv_image = self.stair_line_detector.visualise(cv_image, _lines)
        cv_image = self.stair_line_detector.visualise(cv_image, lines, color=(255, 0, 0))

        proj_lines = self._get_proj_lines(line_iterator, K)
        image_dummy = np.zeros(shape=cv_image.shape, dtype=np.uint8)
        image_dummy[:] = (0, 0, 0)
        for i in range(proj_lines.shape[0]):
            cv2.circle(
                image_dummy,
                (int(proj_lines[i, 0]), int(proj_lines[i, 1])),
                radius=0, color=(255, 255, 255), thickness=-1,
            )
        # (rows, cols, channels) = cv_image.shape
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
        try:
            self.tmp_pub.publish(self.bridge.cv2_to_imgmsg(image_dummy, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def _get_proj_lines(self, line_iterator, K):
        """
        Obtain lines projected onto xy plane from a line_iterator.

        Inputs:
            - line_iterator: line iterator from `N` lines on an image.
                - Size: N x 3
                - line_iterator[i, :] = (u, v, d)
                    - (u, v, 1): homogenous coordinates [1]
                    - d: depth [1]
            - K: intrinsic camera matrix [1, 3]
                - Size: 3x3
        Outputs:
            - proj_lines: the corresponding projected lines.
                - Size: N x 3
                - proj_lines[i, :] = (x, y) for point cloud (x, y, z) in 3D space [1, 2]
        Notes:
            - The formula is based on [Eq. (2), 1].
            - The frame defined in 3D space for point cloud corresponds to homogenous coordinates, see [Fig. 1a, 2].
        Refs:
            [1] I. Vasiljevic et al., “Neural Ray Surfaces for Self-Supervised Learning of Depth and Ego-motion.” arXiv, Aug. 14, 2020. Accessed: Aug. 24, 2022. [Online]. Available: http://arxiv.org/abs/2008.06630
            [2] D. Rosebrock and F. M. Wahl, “Generic camera calibration and modeling using spline surfaces,” in 2012 IEEE Intelligent Vehicles Symposium, Alcal de Henares , Madrid, Spain, Jun. 2012, pp. 51–56. doi: 10.1109/IVS.2012.6232156.
            [3] http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        """
        N = line_iterator.shape[0]
        proj_lines = np.empty(shape=(line_iterator.shape[0], 2))
        proj_lines.fill(np.nan)
        K_inv = np.linalg.inv(K)
        for i in range(N):
            u, v, d = line_iterator[i, :]
            p_tilde = np.array([u, v, 1.0])
            P = d * K_inv @ p_tilde
            proj_lines[i, :] = P[0:2]  # assumption: camera frame's roll&pitch are zero
        return proj_lines

    # def _project_onto_plane(self, point, normal):
    #     """
    #     Project a vector `point` in 3D space
    #     to a plane whose normal vector is `normal`.
    #     Variables:
    #         point = [x, y, z]
    #         normal = [u, v, w]
    #     Notes:
    #         - The plane contains the origin with normal vector of `normal`.
    #         - This is for yaw angle estimation (see ROS app of this class).
    #     Refs:
    #         https://en.wikipedia.org/wiki/Vector_projection
    #     """
    #     normal_unit = normal / np.linalg.norm(normal)
    #     vertical_component = np.dot(point, normal_unit) * normal_unit
    #     return (point - vertical_component)
    #
    def create_line_iterator(self, P1, P2, img):
        """
        Produces an array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x, y)
            -P2: a numpy array that consists of the coordinate of the second point (x, y)
            -img: the image being processed
        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x, y, intensity])
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
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        # itbuffer = np.empty(shape=(np.maximum(dYa, dXa)+1, 3), dtype=np.float32)
        itbuffer.fill(np.nan)

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
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
        return itbuffer


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
