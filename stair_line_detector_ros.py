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
from stair_line_detector import StairLineDetector


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
        (rows, cols, channels) = cv_color.shape

        # lines = self.stair_line_detector.detect_lines(cv_color)
        # cv_color = self.stair_line_detector.visualise(cv_color, lines, color=(255, 0, 0))

        _lines = self.stair_line_detector._detect_lines(cv_color)
        lines = self.stair_line_detector._filter_outlier_out(_lines)
        lines = [lines[-1]]  # TODO: change this
        line_iterators = [
            self.create_line_iterator(line[0:2], line[2:4], cv_color)
            for line in lines
        ]
        # for li in line_iterators:
        #     inflated_lis = []
        #     for inflation in [
        #             [+3, 0], [+2, 0], [+1, 0], [-1, 0], [-2, 0], [-3, 0],
        #             [0, +3], [0, +2], [0, +1], [0, -1], [0, -2], [0, -3],
        #             [+2, +1], [+2, +2], [+1, +2], [+1, +1],
        #             [+2, -1], [+2, -2], [+1, -2], [+1, -1],
        #             [-2, +1], [-2, +2], [-1, +2], [-1, +1],
        #             [-2, -1], [-2, -2], [-1, -2], [-1, -1],
        #     ]:
        #         inflated_lis.append([list(inflated) for inflated in np.array(li)+inflation])
        #     for inflated_li in inflated_lis:
        #         li = li + inflated_li
        # # TODO: remove it
        # line_iterators = [[]]
        # for i in range(int(rows/4)):
        #     for j in range(int(cols/4)):
        #         line_iterators[0].append([i, j])
        # # TODO: remove it
        points_iterators = [
            read_points(
                pcd,
                field_names=["x", "y", "z"],
                uvs=line_iterator,
                # uvs=[[p[0], p[1]] for p in line_iterator],
                skip_nans=True,
            )
            for line_iterator in line_iterators
        ]
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
            # points_all = list(set(points_all))
            pcd_all = create_cloud_xyz32(pcd.header, points_all)
            # pcd_all = create_cloud(pcd.header, pcd.fields, points_all)
            self.pcd_pub.publish(pcd_all)
        import pdb; pdb.set_trace()

    # def _get_point_cloud(self, line_iterator, K):
    #     """
    #     Obtain point cloud from line_iterator with depth data.
    #
    #     Inputs:
    #         - line_iterator: line iterator from `N` lines on an image.
    #             - Size: N x 3
    #             - line_iterator[i, :] = (u, v, d)
    #                 - (u, v, 1): homogenous coordinates [1]
    #                 - d: depth [1]
    #         - K: intrinsic camera matrix [1, 3]
    #             - Size: 3x3
    #     Outputs:
    #         - pcd: obtained point cloud
    #             - Size: N x 3
    #             - pcd[i, :] = (x, y, z) in 3D space [1, 2]
    #     Notes:
    #         - The formula is based on [Eq. (2), 1].
    #         - The frame defined in 3D space for point cloud corresponds to homogenous coordinates, see [Fig. 1a, 2].
    #     Refs:
    #         [1] I. Vasiljevic et al., “Neural Ray Surfaces for Self-Supervised Learning of Depth and Ego-motion.” arXiv, Aug. 14, 2020. Accessed: Aug. 24, 2022. [Online]. Available: http://arxiv.org/abs/2008.06630
    #         [2] D. Rosebrock and F. M. Wahl, “Generic camera calibration and modeling using spline surfaces,” in 2012 IEEE Intelligent Vehicles Symposium, Alcal de Henares , Madrid, Spain, Jun. 2012, pp. 51–56. doi: 10.1109/IVS.2012.6232156.
    #         [3] http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    #     """
    #     N = line_iterator.shape[0]
    #     pcd = np.empty(shape=(line_iterator.shape[0], 3))
    #     pcd.fill(np.nan)
    #     K_inv = np.linalg.inv(K)
    #     for i in range(N):
    #         u, v, d = line_iterator[i, :]
    #         p_tilde = np.array([u, v, 1.0])
    #         P = d * K_inv @ p_tilde
    #         pcd[i, :] = P  # assumption: camera frame's roll&pitch are zero
    #     return pcd
    #
    # def _get_proj_lines(self, line_iterator, K):
    #     """
    #     Obtain lines projected onto xy plane from a line_iterator.
    #
    #     Assumption:
    #         - [A1] roll/pitch of the camera frame w.r.t. fixed frame are zero.
    #     """
    #     pcd = self._get_point_cloud(line_iterator, K)
    #     proj_lines = pcd[:, 0:2]  # (x, y, z) -> (x, y) [A1]
    #     return proj_lines

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
    def create_line_iterator(self, P1, P2, img, stride=None):
        # NOTE: line should be from the original img, not (512, 512).
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
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2), dtype=np.uint16)
        # itbuffer = np.empty(shape=(np.maximum(dYa, dXa)+1, 3), dtype=np.float32)
        # itbuffer.fill(np.nan)

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
        if stride is not None:
            line_iterator = self._inflate_line_iterator(line_iterator, stride)
        return line_iterator

    def _inflate_line_iterator(self, line_iterator, stride: int = 3):
        """
        NOTE: it takes too long time
        """
        if stride <= 0.0 or stride % 2 != 1:
            raise ValueError("Invalid inflation stride")
        else:
            for index in line_iterator:
                for _i in range(stride):
                    for _j in range(stride):
                        i = _i - int((stride-1)/2)
                        j = _j - int((stride-1)/2)
                        new_index = [index[0]+i, index[1]+j]
                        if new_index not in line_iterator:
                            line_iterator.append(new_index)
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
