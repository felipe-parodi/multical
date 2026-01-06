from functools import partial, reduce
import operator
from cached_property import cached_property
import numpy as np
import cv2
from structs.numpy import shape

from structs.struct import subset, transpose_structs, transpose_lists

from pprint import pformat

from .transform import rtvec, matrix

from structs.struct import struct
from .optimization.parameters import Parameters

from multiprocessing.pool import ThreadPool
from multical.threading import cpu_count

import cv2
from tqdm import tqdm

from structs.struct import split_list

# from .optimization.multiscale import multiscale_calibrate


class Camera(Parameters):
    def __init__(
        self,
        image_size,
        intrinsic,
        dist,
        model="standard",
        fix_aspect=False,
        has_skew=False,
        error_perview=None,
        intrinsic_dataset={},
    ):

        assert (
            model in Camera.model
        ), f"unknown camera model {model} options are {list(self.model.keys())}"

        self.model = model

        self.image_size = tuple(image_size)
        self.intrinsic = intrinsic
        self.dist = np.zeros(5) if dist is None else dist
        self.fix_aspect = fix_aspect
        self.has_skew = has_skew
        self.error_perview = error_perview  #
        self.intrinsic_dataset = (
            intrinsic_dataset  # Collects views that are used for intrinsic calibration
        )

    model = struct(
        standard=0,
        rational=cv2.CALIB_RATIONAL_MODEL,
        tilted=cv2.CALIB_TILTED_MODEL,
        thin_prism=cv2.CALIB_THIN_PRISM_MODEL,
        full=cv2.CALIB_RATIONAL_MODEL
        + cv2.CALIB_THIN_PRISM_MODEL
        + cv2.CALIB_TILTED_MODEL,
    )

    def __str__(self):
        d = dict(intrinsic=self.intrinsic, dist=self.dist, image_size=self.image_size)
        return "Camera " + pformat(d)

    def __repr__(self):
        return self.__str__()

    def approx_eq(self, other):
        assert isinstance(other, Camera)
        return (
            self.image_size == other.image_size
            and np.allclose(other.intrinsic, self.intrinsic)
            and np.allclose(other.dist, self.dist)
        )

    @staticmethod
    def flags(model, fix_aspect=False):
        return Camera.model[model] | cv2.CALIB_FIX_ASPECT_RATIO * fix_aspect

    @staticmethod
    def calibrate(
        boards,
        intrinsic_error_limit,
        detections,
        image_size,
        max_iter=10,
        eps=1e-3,
        model="standard",
        fix_aspect=False,
        has_skew=False,
        flags=0,
        max_images=None,
        fix_radial=False,
        fix_tangential=True,
    ):
        # def calibrate(boards, detections, image_size, intrinsic_error_limit,
        #               max_iter=10, eps=1e-3, model='standard', fix_aspect=False,
        #               has_skew=False, flags=0, max_images=None, use_multiscale=False,
        #               scales=[0.25, 0.5, 1.0], **kwargs):
        """
        iteratively selects best images to calculate intrinsic parameters
        """
        # if use_multiscale:
        #     return multiscale_calibrate(
        #         boards=boards,
        #         detections=detections,
        #         image_size=image_size,
        #         intrinsic_error_limit=intrinsic_error_limit,
        #         max_iter=max_iter,
        #         eps=eps,
        #         model=model,
        #         fix_aspect=fix_aspect,
        #         has_skew=has_skew,
        #         flags=flags,
        #         max_images=max_images,
        #         scales=scales,
        #         **kwargs
        #     )
        points = calibration_points(boards, detections)
        if max_images is not None:
            points = top_detection_coverage(points, max_images, image_size)

        # Validate that we have images to calibrate with
        num_images = len(points.corners) if hasattr(points, 'corners') else 0
        if num_images == 0:
            raise ValueError(
                f"No valid calibration images found for this camera. "
                f"Ensure the calibration board is visible in at least some images. "
                f"Image size: {image_size}"
            )

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        flags = Camera.flags(model, fix_aspect) | flags

        # Configurable distortion flags
        if fix_radial:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        if fix_tangential:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST

        # Log which distortion model is being used
        radial_status = "fixed (k1=k2=k3=0)" if fix_radial else "free (estimating k1,k2,k3)"
        tangent_status = "fixed (p1=p2=0)" if fix_tangential else "free (estimating p1,p2)"
        print(f"Distortion model: radial={radial_status}, tangential={tangent_status}")

        err = intrinsic_error_limit
        while abs(err) >= intrinsic_error_limit:
            err, K, dist, r, t, _, _, error_perView = cv2.calibrateCameraExtended(
                points.object_points,
                points.corners,
                image_size,
                None,
                None,
                criteria=criteria,
                flags=flags,
            )
            if len(error_perView) >= 15:
                err = float("{:.2f}".format(err))
                threshold = np.quantile(error_perView, 0.95)
                inliers = [
                    (i) for i, err in enumerate(error_perView) if err < threshold
                ]
                points.object_points = np.array(
                    [points.object_points[i] for i in inliers], dtype=object
                )
                points.corners = np.array(
                    [points.corners[i] for i in inliers], dtype=object
                )
                points.ids = np.array([points.ids[i] for i in inliers], dtype=object)
                points.board_offset = np.array(
                    [points.board_offset[i] for i in inliers], dtype=object
                )
                points.image_ids = np.array(
                    [points.image_ids[i] for i in inliers], dtype=object
                )
            else:
                intrinsic_error_limit += 0.1

        return (
            Camera(
                intrinsic=K,
                dist=dist,
                image_size=image_size,
                model=model,
                fix_aspect=fix_aspect,
                has_skew=has_skew,
                error_perview=error_perView,
                intrinsic_dataset={
                    "board_ids": list(points.board_offset),
                    "image_ids": list(points.image_ids),
                },
            ),
            err,
        )

    def scale_image(self, factor):
        intrinsic = self.intrinsic.copy()
        intrinsic[:2] *= factor

        return self.copy(intrinsic=intrinsic)

    @cached_property
    def undistort_map(self):
        m, _ = cv2.initUndistortRectifyMap(
            self.intrinsic,
            self.dist,
            None,
            self.intrinsic,
            self.image_size,
            cv2.CV_32FC2,
        )
        return m

    def undistort_points(self, points):
        undistorted = cv2.undistortPoints(
            points.reshape(-1, 1, 2), self.intrinsic, self.dist, P=self.intrinsic
        )
        return undistorted.reshape(*points.shape[:-1], 2)

    def project(self, points):

        projected, _ = cv2.projectPoints(
            cv2.UMat(points.reshape(-1, 1, 3)),
            np.zeros(3),
            np.zeros(3),
            self.intrinsic,
            self.dist,
        )
        return projected.get().reshape(*points.shape[:-1], 2)

    @cached_property
    def focal_length(self):
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        return np.array([fx, fy])

    @cached_property
    def principle_point(self):
        return np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]])

    @cached_property
    def skew(self):
        return self.intrinsic[0, 1] if self.has_skew else 0.0

    @cached_property
    def params(self):
        f = self.focal_length
        if self.fix_aspect:
            f = np.array([f.mean(), f.mean()])

        return struct(
            focal_length=f,
            principle_point=self.principle_point,
            skew=np.array([self.skew]),
            dist=self.dist,
        )

    def with_params(self, params):

        f = params.focal_length
        fx, fy = f if not self.fix_aspect else (f[0], f[0])

        px, py = params.principle_point
        (skew,) = params.skew

        intrinsic = [
            [fx, skew, px],
            [0, fy, py],
            [0, 0, 1],
        ]

        return self.copy(intrinsic=np.array(intrinsic), dist=params.dist)

    def __getstate__(self):
        return subset(
            self.__dict__,
            ["image_size", "intrinsic", "dist", "fix_aspect", "has_skew", "model"],
        )

    def copy(self, **k):
        d = self.__getstate__()
        d.update(k)
        return Camera(**d)


def board_correspondences(board_id, board, detections):
    non_empty = [d for d in detections if board.has_min_detections(d)]
    img_ids = [id for id, d in enumerate(detections) if board.has_min_detections(d)]
    if len(non_empty) == 0:
        return struct(
            corners=[], object_points=[], ids=[], board_offset=[], image_ids=[]
        )

    detections = transpose_structs(non_empty)
    return detections._extend(
        object_points=[board.points[ids].astype(np.float32) for ids in detections.ids],
        corners=[corners.astype(np.float32) for corners in detections.corners],
        board_offset=list(np.ones(len(img_ids)) * board_id),
        image_ids=img_ids,
    )


def board_frames(board, detections):
    non_empty = [d for d in detections if board.has_min_detections(d)]
    return len(non_empty)


def index_list(xs, indexes):
    return np.array(xs, dtype=object)[indexes].tolist()


def coverage(corners, bins):
    hist, _, _ = np.histogram2d(corners[:, 0], corners[:, 1], bins)
    counts = np.count_nonzero(hist)

    return counts


def image_bins(image_size, approx_bins=10):
    bin_size = min(image_size[0] / approx_bins, image_size[1] / approx_bins)

    return [
        np.linspace(0, image_size[axis], int(image_size[axis] / bin_size))
        for axis in [0, 1]
    ]


def top_detection_coverage(detections, k, image_size, approx_bins=10, jitter=0.1):
    bins = image_bins(image_size, approx_bins=10)
    bin_jitter = jitter * (approx_bins * approx_bins)

    sizes = [
        -coverage(corners, bins) + np.random.normal(0, bin_jitter)
        for corners in detections.corners
    ]

    sorted = detections._map(index_list, np.argsort(sizes))
    return sorted._map(lambda xs: xs[:k])


def calibration_points(boards, detections):

    board_detections = transpose_lists(detections)
    board_points = [
        board_correspondences(board_id, board, detections)
        for board_id, (board, detections) in enumerate(zip(boards, board_detections))
    ]

    return reduce(operator.add, board_points)


def calibrate_cameras(boards, points, image_sizes, intrinsic_error_limit, camera_names=None, **kwargs):

    # Pre-validate that each camera has detections before multiprocessing
    # Check all cameras first and collect failures
    camera_stats = []
    failed_cameras = []

    for i, (cam_points, img_size) in enumerate(zip(points, image_sizes)):
        cam_calib_points = calibration_points(boards, cam_points)
        num_images = len(cam_calib_points.corners) if hasattr(cam_calib_points, 'corners') else 0
        camera_id = camera_names[i] if camera_names and i < len(camera_names) else f"Camera {i}"

        camera_stats.append((camera_id, num_images, img_size))

        if num_images == 0:
            failed_cameras.append(camera_id)

    # Print summary of all cameras
    print("\nCalibration image count per camera:")
    for camera_id, num_images, img_size in camera_stats:
        status = "✓" if num_images > 0 else "✗"
        print(f"  {status} {camera_id}: {num_images} valid calibration images")

    # If any cameras failed, report them all at once
    if failed_cameras:
        raise ValueError(
            f"\n{'='*70}\n"
            f"ERROR: {len(failed_cameras)} camera(s) have NO valid board detections!\n"
            f"{'='*70}\n"
            f"Failed cameras: {', '.join(failed_cameras)}\n"
            f"\n"
            f"These cameras need images with the calibration board visible.\n"
            f"Check that:\n"
            f"  - Images for these cameras contain the calibration board\n"
            f"  - The board pattern matches the board configuration file\n"
            f"  - Images are not corrupted or too blurry\n"
            f"  - Camera names/folders are correct\n"
            f"{'='*70}"
        )

    with ThreadPool() as pool:
        f = partial(Camera.calibrate, boards, intrinsic_error_limit, **kwargs)
        return transpose_lists(pool.starmap(f, zip(points, image_sizes)))


def undistort_image(args):
    image, undistort_map = args
    return cv2.remap(image, undistort_map, None, cv2.INTER_CUBIC)


def undistort_images(images, cameras, j=cpu_count(), chunksize=4):
    with ThreadPool(processes=j) as pool:
        image_pairs = [
            (image, camera.undistort_map)
            for camera, cam_images in zip(cameras, images)
            for image in cam_images
        ]

        loader = pool.imap(undistort_image, image_pairs, chunksize=chunksize)
        undistorted = list(tqdm(loader, total=len(image_pairs)))

        return split_list(undistorted, [len(i) for i in images])


def stereo_calibrate(
    cameras, matches, max_iter=60, eps=1e-6, fix_aspect=False, fix_intrinsic=True
):

    left, right = cameras
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)

    assert left.image_size == right.image_size
    assert left.model == right.model

    model = left.model
    image_size = left.image_size

    flags = (
        Camera.flags(model, fix_aspect)
        | cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_INTRINSIC * fix_intrinsic
    )

    err, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        matches.object_points,
        matches.points1,
        matches.points2,
        left.intrinsic,
        left.dist,
        right.intrinsic,
        right.dist,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    left = Camera(dist=d1, intrinsic=K1, image_size=image_size, model=model)
    right = Camera(dist=d2, intrinsic=K2, image_size=image_size, model=model)

    return left, right, matrix.join(R, T.flatten()), err
