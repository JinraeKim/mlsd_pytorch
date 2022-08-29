#!/usr/bin/env python3
from copy import copy
import cv2

import numpy as np

# with M-LSD
from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
# from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import pred_lines
import torch

from numpy.random import default_rng


class RANSAC:
    """
    Borrowed from https://en.wikipedia.org/wiki/Random_sample_consensus#Example_Code.
    The computation time will be proportional to `k`.
    """
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: class implementing `fit` and `predict`
        self.loss = loss        # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric    # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf
        self.inlier_indices = None
        self.rng = default_rng()

    def fit(self, X, y):
        if len(X) == 0:
            pass
        else:
            for _ in range(self.k):
                ids = self.rng.permutation(X.shape[0])
                n = self.n

                maybe_inliers = ids[:n]
                maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

                thresholded = (
                    self.loss(y[ids][n:], maybe_model.predict(X[ids][n:]))
                    < self.t
                )
                inlier_ids = ids[n:][np.flatnonzero(thresholded).flatten()]

                if inlier_ids.size > self.d:
                    inlier_ids_including_maybe = np.hstack([maybe_inliers, inlier_ids])
                    better_model = copy(self.model).fit(X[inlier_ids_including_maybe], y[inlier_ids_including_maybe])

                    this_error = self.metric(
                        y[inlier_ids_including_maybe], better_model.predict(X[inlier_ids_including_maybe])
                    )

                    if this_error < self.best_error:
                        self.best_error = this_error
                        self.best_fit = maybe_model
                        self.inlier_indices = inlier_ids
        return self

    def predict(self, X):
        return self.best_fit.predict(X)


def square_error_loss(y_true, y_pred):
    return wraptopi(y_true - y_pred) ** 2  # speicialised to deal with angles


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


def wraptopi(angles):
    """
    Convert angles into [-pi, pi)
    """
    return (angles + np.pi) % (2*np.pi) - np.pi


class LinearRegressor:
    def __init__(self):
        self.params = None
        self.params_dim = (1+1, 1)  # in case of receiving no data; +1 is for bias
        self.success = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        try:
            self.params = np.linalg.inv(X.T @ X) @ X.T @ y
            self.success = True
        except np.linalg.LinAlgError as e:
            self.params = np.random.normal(size=np.prod(self.params_dim)).reshape(self.params_dim)
            print("maybe too small number of lines are provided, random parameter is generated;", e)
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


class StairLineDetector:
    def __init__(self):
        """
        `regressor` is a RANSAC regressor.
        """
        model_path = "." + '/models/mlsd_tiny_512_fp32.pth'
        model = MobileV2_MLSD_Tiny().cuda().eval()
        # model_path = "." +'/models/mlsd_large_512_fp32.pth'
        # model = MobileV2_MLSD_Large().cuda().eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        self.model = model
        self.img_size_for_inference = (512, 512)

    def _detect_lines(self, img):
        """
        img: opencv color img
        lines: predicted lines.
        """
        lines = pred_lines(img, self.model, self.img_size_for_inference, 0.1, 100)
        return lines

    def _filter_outlier_out(self, lines):
        thetas = np.array([
            wraptopi(np.arctan2(l[3]-l[1], l[2]-l[0]) - 0.5*np.pi)
            for l in lines
        ]).reshape(-1, 1)
        rs = np.array([
            l[0]*np.cos(theta) + l[1]*np.sin(theta)
            for l, theta in zip(lines, thetas)
        ]).reshape(-1, 1)

        # update regressor (adaptive inlier setting)
        n = int(rs.shape[0] / 4)  # parameter for the number of inliers
        d = int(rs.shape[0] / 4)  # parameter for the number of inliers
        regressor = RANSAC(
            n=n, d=d,
            model=LinearRegressor(),
            loss=square_error_loss,
            metric=mean_square_error,
        )
        # fitting regressor model
        regressor.fit(rs, thetas)

        if regressor.inlier_indices is None:
            lines_inlier = lines
        else:
            lines_inlier = lines[regressor.inlier_indices]
        return lines_inlier

    def detect_lines(self, img):
        lines = self._detect_lines(img)
        return self._filter_outlier_out(lines)

    def visualise(self, img, lines, color=(0, 0, 255)):
        for l in lines:
            cv2.line(
                img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
                color, 2,
            )
        return img


if __name__ == "__main__":
    """
    Author:
        Jinrae Kim
    Basic idea:
        When lines are detected, one can represent them with
        `r` and `theta` (Polar coordinate).
        If the lines are obtained from stairs, there may be many parallel lines.
        Then, there would be a command `theta` among those lines.
        Note that RANSAC is applied to get a line (instead of a single parameter
        of `theta` here.
    """
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    file = args.file
    img = cv2.imread(file)

    detector = StairLineDetector()
    t0 = time.time()
    lines = detector._detect_lines(img)
    t1 = time.time()
    print(f"_detect_lines time: {t1-t0}s")

    t0 = time.time()
    lines_inlier = detector._filter_outlier_out(lines)
    t1 = time.time()
    print(f"_filter_outlier_out time: {t1-t0}s")
    # _, lines = detector.find_edges_and_lines(img)
    img = detector.visualise(img, lines)
    img = detector.visualise(img, lines_inlier, color=(255, 0, 0))
    cv2.imwrite(file[:-4] + "_with_lines" + file[-4:], img)
    # lines = detector.postprocess_lines(lines)

    thetas = np.array([
        wraptopi(np.arctan2(l[3]-l[1], l[2]-l[0]) - 0.5*np.pi)
        for l in lines
    ]).reshape(-1, 1)
    rs = np.array([
        l[0]*np.cos(theta) + l[1]*np.sin(theta)
        for l, theta in zip(lines, thetas)
    ]).reshape(-1, 1)

    # regression
    n = int(rs.shape[0] / 4)  # parameter for the number of inliers
    d = int(rs.shape[0] / 4)  # parameter for the number of inliers
    regressor = RANSAC(n=n, d=d, model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
    X, y = rs, thetas
    regressor.fit(X, y)

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    plt.scatter(X, y)
    plt.scatter(X[regressor.inlier_indices], y[regressor.inlier_indices])

    line = np.linspace(np.min(X), np.max(X), num=100).reshape(-1, 1)
    # print(f"The best params are: {regressor.best_fit.params}")
    thetas_readable = wraptopi(-(thetas + 0.5*np.pi))  # human readable; x, y for left to right and down to up, resp.
    print(f"The predicted stair angle is: {np.mean(np.rad2deg(thetas_readable[regressor.inlier_indices]))} [deg]")
    plt.plot(line, regressor.predict(line), c="peru")
    plt.xlabel("r")
    plt.ylabel("theta (related to line slopes)")
    plt.savefig("inlier_outlier.png")
