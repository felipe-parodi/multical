from collections import OrderedDict
from multical.threading import parmap_lists
import pathlib
from multical.board.board import Board

import numpy as np
from multical.motion import StaticFrames
from multiprocessing import cpu_count

from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical import config

from os import path
from multical.io import export_json, try_load_detections, write_detections
from multical.image.detect import common_image_size

from multical.optimization.calibration import Calibration, select_threshold
from structs.struct import map_list, split_dict, struct, subset, to_dicts
from . import tables, image
from .camera import calibrate_cameras
from .hand_eye.hand_eye import *

from structs.numpy import shape
from structs.numpy import Table

from .camera_fisheye import calibrate_cameras_fisheye
from .io.logging import MemoryHandler, info
from .display import color_sets
from .io.detections import try_load_cache_data

import pickle
import json


def detect_boards_cached(
    boards, images, detections_file, cache_key, load_cache=True, j=cpu_count()
):
    assert isinstance(boards, list)

    # First try exact cache match
    detected_points = (
        try_load_detections(detections_file, cache_key) if load_cache else None
    )

    # If exact match fails, try partial cache reuse
    if detected_points is None and load_cache:
        detected_points = detect_boards_with_partial_cache(
            boards, images, detections_file, cache_key, j=j
        )

    # If still no detections, detect all
    if detected_points is None:
        info("Detecting boards..")
        detected_points = image.detect.detect_images(boards, images, j=j)

        info(f"Writing detection cache to {detections_file}")
        write_detections(detections_file, detected_points, cache_key)

    return detected_points


def detect_boards_with_partial_cache(boards, images, detections_file, cache_key, j=cpu_count()):
    """
    Try to reuse cached detections for cameras that exist in both cache and current run.
    Only detect boards for new cameras.
    """
    cached_data = try_load_cache_data(detections_file)
    if cached_data is None:
        return None

    cached_key = cached_data.get("cache_key", {})

    # Check if cache has camera_names
    current_camera_names = cache_key.get("camera_names", [])
    cached_camera_names = cached_key.get("camera_names", [])

    if not current_camera_names or not cached_camera_names:
        info("Cache doesn't include camera names - cannot do partial reuse")
        return None

    # Check if boards config matches
    if cached_key.get("boards") != cache_key.get("boards"):
        info("Board configuration changed - cannot reuse cache")
        return None

    # Match cameras by name
    cached_detections = cached_data.get("detected_points", [])
    cached_filenames = cached_key.get("filenames", [])
    cached_image_sizes = cached_key.get("image_sizes", [])

    current_filenames = cache_key.get("filenames", [])
    current_image_sizes = cache_key.get("image_sizes", [])

    # Build lookup dict for cached data
    cache_lookup = {}
    for cam_name, detections, filenames, img_size in zip(
        cached_camera_names, cached_detections, cached_filenames, cached_image_sizes
    ):
        cache_lookup[cam_name] = {
            'detections': detections,
            'filenames': filenames,
            'image_size': img_size
        }

    # Determine which cameras we can reuse and which need detection
    cameras_to_detect = []
    cameras_to_detect_indices = []
    reused_count = 0
    detect_count = 0

    result_detections = []

    for i, (cam_name, filenames, img_size, imgs) in enumerate(zip(
        current_camera_names, current_filenames, current_image_sizes, images
    )):
        if cam_name in cache_lookup:
            cached = cache_lookup[cam_name]
            # Check if filenames and image size match
            if (cached['filenames'] == filenames and
                cached['image_size'] == img_size):
                # Reuse cached detections
                result_detections.append(cached['detections'])
                reused_count += 1
                info(f"  Reusing cached detections for {cam_name}")
            else:
                # Need to re-detect (images changed)
                cameras_to_detect.append(imgs)
                cameras_to_detect_indices.append(i)
                result_detections.append(None)  # Placeholder
                detect_count += 1
        else:
            # New camera not in cache
            cameras_to_detect.append(imgs)
            cameras_to_detect_indices.append(i)
            result_detections.append(None)  # Placeholder
            detect_count += 1

    if detect_count == 0:
        info(f"Reused all {reused_count} cameras from cache")
        return result_detections

    if reused_count > 0:
        info(f"Reusing {reused_count} cameras from cache, detecting {detect_count} cameras")

        # Detect boards for cameras that need it
        info("Detecting boards for new/changed cameras..")
        new_detections = image.detect.detect_images(boards, cameras_to_detect, j=j)

        # Merge new detections into result
        for idx, detections in zip(cameras_to_detect_indices, new_detections):
            result_detections[idx] = detections

        # Write updated cache
        info(f"Writing updated detection cache to {detections_file}")
        write_detections(detections_file, result_detections, cache_key)

        return result_detections

    # No cameras could be reused
    return None


def num_valid_detections(boards, frames):
    n = 0
    for frame_detections in frames:
        for board, dets in zip(boards, frame_detections):
            if board.has_min_detections(dets):
                n = n + 1
    return n


def check_detections(camera_names, boards, detected_points):
    cameras = [
        k
        for k, fame_detections in zip(camera_names, detected_points)
        if num_valid_detections(boards, fame_detections) == 0
    ]

    assert (
        len(cameras) == 0
    ), f"cameras {cameras} have no valid detections, check board config"


def check_image_lengths(cameras, filenames, image_names):
    for k, images in zip(cameras, filenames):
        assert len(images) == len(image_names), (
            f"mismatch between image names and camera {k}, "
            f"got {len(images)} filenames expected {len(image_names)}"
        )


def check_camera_images(camera_images):
    assert len(camera_images.cameras) == len(camera_images.filenames), (
        f"expected filenames to be a list of equal to number of cameras "
        f"{len(camera_images.cameras)} vs. {len(camera_images.filenames)}"
    )

    check_image_lengths(
        camera_images.cameras, camera_images.filenames, camera_images.image_names
    )
    if "images" in camera_images is not None:
        check_image_lengths(
            camera_images.cameras, camera_images.images, camera_images.image_names
        )


class Workspace:

    def __init__(self, output_path, name="calibration"):

        self.name = name
        self.output_path = output_path

        self.calibrations = OrderedDict()
        self.detections = None
        self.boards = None
        self.board_colors = None

        self.filenames = None
        self.image_path = None
        self.names = struct()

        self.image_sizes = None
        self.images = None

        self.point_table = None
        self.pose_table = None

        self.log_handler = MemoryHandler()

    def add_camera_images(self, camera_images, j=cpu_count()):
        check_camera_images(camera_images)
        self.names = self.names._extend(
            camera=camera_images.cameras, image=camera_images.image_names
        )

        self.filenames = camera_images.filenames
        self.image_path = camera_images.image_path

        if "images" in camera_images:
            self.images = camera_images.images
            self.image_size = map_list(common_image_size, self.images)
        else:
            self._load_images(j=j)

    def _load_images(self, j=cpu_count()):
        assert self.filenames is not None, "_load_images: no filenames set"

        info("Loading images..")
        self.images = image.detect.load_images(
            self.filenames, j=j, prefix=self.image_path
        )
        self.image_size = map_list(common_image_size, self.images)

        info(f"Loaded {self.sizes.image * self.sizes.camera} images")
        info(
            {k: image_size for k, image_size in zip(self.names.camera, self.image_size)}
        )

    @property
    def detections_file(self):
        return path.join(self.output_path, f"{self.name}.detections.pkl")

    def detect_boards(self, boards, load_cache=True, j=cpu_count()):
        assert self.boards is None, "detect_boards: boards already set"
        assert (
            self.images is not None
        ), "detect_boards: no images loaded, first use add_camera_images"

        board_names, self.boards = split_dict(boards)
        self.names = self.names._extend(board=board_names)
        self.board_colors = color_sets["set1"]
        cache_key = self.fields("filenames", "boards", "image_sizes")

        self.detected_points = detect_boards_cached(
            self.boards, self.images, self.detections_file, cache_key, load_cache, j=j
        )

        self.point_table = tables.make_point_table(self.detected_points, self.boards)
        info("Detected point counts:")
        tables.table_info(self.point_table.valid, self.names)

    def set_calibration(self, cameras):
        """
        Set camera calibration from a dictionary of camera objects.
        Cameras in the calibration file but not in current images will be ignored.
        Cameras in current images but not in calibration file will raise an error.

        Args:
          cameras: Dictionary mapping camera names to Camera objects
        """
        # Find which cameras are in both sets, and which are missing
        current_cameras = set(self.names.camera)
        calib_cameras = set(cameras.keys())

        available_cameras = current_cameras & calib_cameras
        missing_in_calib = current_cameras - calib_cameras
        extra_in_calib = calib_cameras - current_cameras

        # Info about mismatches
        if extra_in_calib:
            info(
                f"Note: {len(extra_in_calib)} camera(s) in calibration file but not in current images (will be ignored):"
            )
            info(f"  {sorted(extra_in_calib)}")
            info("")

        if missing_in_calib:
            info(
                f"ERROR: {len(missing_in_calib)} camera(s) in current images but not in calibration file:"
            )
            info(f"  {sorted(missing_in_calib)}")
            info("")
            raise ValueError(
                f"Cannot use calibration file: missing intrinsics for {sorted(missing_in_calib)}.\n"
                + f"Options:\n"
                + f"  (1) Exclude these cameras from your image set, or\n"
                + f"  (2) Remove --calibration flag to calibrate all cameras from scratch, or\n"
                + f"  (3) Add intrinsic calibration for these cameras to your calibration file."
            )

        assert (
            len(available_cameras) > 0
        ), f"set_calibration: no cameras in common between images and calibration file"

        # Use only cameras that are in current image set (filtering out extras)
        self.cameras = [cameras[k] for k in self.names.camera]

        info(f"Loaded calibration for {len(self.cameras)} camera(s)")
        for name, camera in zip(self.names.camera, self.cameras):
            info(f"{name} {camera}")
            info("")

    def calibrate_single(
        self,
        camera_model,
        intrinsic_error_limit,
        fix_aspect=False,
        has_skew=False,
        max_images=None,
        isFisheye=False,
    ):
        assert (
            self.detected_points is not None
        ), "calibrate_single: no points found, first use detect_boards to find corner points"

        check_detections(self.names.camera, self.boards, self.detected_points)

        info("Calibrating single cameras..")
        if not isFisheye:
            self.cameras, errs = calibrate_cameras(
                self.boards,
                self.detected_points,
                self.image_size,
                intrinsic_error_limit,
                model=camera_model,
                fix_aspect=fix_aspect,
                has_skew=has_skew,
                max_images=max_images,
            )
        else:
            self.cameras, errs = calibrate_cameras_fisheye(
                self.boards,
                self.detected_points,
                self.image_size,
                model=camera_model,
                fix_aspect=fix_aspect,
                has_skew=has_skew,
                max_images=max_images,
            )

        for name, camera, err in zip(self.names.camera, self.cameras, errs):
            info(f"Calibrated {name}, with RMS={err:.2f}")
            info(camera)
            info("")

    def initialise_poses(
        self,
        motion_model=StaticFrames,
        camera_poses=None,
        exclude_bad_poses=True,
        pose_error_limit=1.0,
        is_non_overlapping=False,
    ):
        assert (
            self.cameras is not None
        ), "initialise_poses: no cameras set, first use calibrate_single or set_cameras"
        self.pose_table = tables.make_pose_table(
            self.point_table,
            self.boards,
            self.cameras,
            exclude_bad_poses,
            pose_error_limit,
        )

        info("Pose counts:")
        tables.table_info(self.pose_table.valid, self.names)

        # Highlight cameras with zero valid poses
        try:
            cam_pose_counts = tables.count_valid(self.pose_table.valid, axes=[0])
            zero_cam_ids = np.where(cam_pose_counts == 0)[0]
            if zero_cam_ids.size > 0:
                zero_cams = [self.names.camera[i] for i in zero_cam_ids]
                warning_msg = (
                    "\n" +
                    "=" * 70 + "\n" +
                    f"WARNING: {len(zero_cams)} camera(s) have 0 valid poses for extrinsics!\n" +
                    "=" * 70 + "\n" +
                    f"Cameras: {', '.join(zero_cams)}\n\n" +
                    "These cameras cannot be assigned an extrinsic pose and will be\n" +
                    "excluded from the exported camera_poses.\n" +
                    "Consider dropping these cameras or recapturing with visible boards.\n" +
                    "=" * 70
                )
                info(warning_msg)
        except Exception:
            # Do not fail calibration due to logging
            pass

        # Non-overlapping case consideration
        if is_non_overlapping and camera_poses is None:
            handeye = HandEye(self.pose_table, self.names.camera, self.image_path)
            handeye.initialise_camera_poses()
            camera_poses = handeye.cam_init

        pose_init = tables.initialise_poses(
            self.pose_table,
            camera_poses=(
                None
                if camera_poses is None
                else np.array([camera_poses[k] for k in self.names.camera])
            ),
        )

        calib = Calibration(
            ParamList(self.cameras, self.names.camera),
            ParamList(self.boards, self.names.board),
            self.point_table,
            PoseSet(pose_init.camera, self.names.camera),
            PoseSet(pose_init.board, self.names.board),
            motion_model.init(pose_init.times, self.names.image),
        )

        # calib = calib.reject_outliers_quantile(0.75, 5)
        calib.report(f"Initialisation")

        self.calibrations["initialisation"] = calib
        return calib

    def calibrate(
        self,
        name="calibration",
        camera_poses=True,
        motion=True,
        board_poses=True,
        cameras=False,
        boards=False,
        loss="linear",
        tolerance=1e-4,
        num_adjustments=3,
        quantile=0.75,
        auto_scale=None,
        outlier_threshold=5.0,
        reject_view_threshold=None,
        ) -> Calibration:

        calib: Calibration = self.latest_calibration.enable(
            cameras=cameras,
            boards=boards,
            camera_poses=camera_poses,
            motion=motion,
            board_poses=board_poses,
        )

        # --- Existing iterative refinement ---
        calib = calib.adjust_outliers(
            loss=loss,
            tolerance=tolerance,
            num_adjustments=num_adjustments,
            select_outliers=select_threshold(
                quantile=quantile, factor=outlier_threshold
            ),
            select_scale=(
                select_threshold(quantile=quantile, factor=auto_scale)
                if auto_scale is not None
                else None
            ),
        )
        calib.report(
            f"After adjust_outliers for '{name}'"
        )  # Report state after adjustments

        # Summarize cameras lacking extrinsic poses
        try:
            cam_valid = calib.camera_poses.valid
            zero_cam_ids = np.where(~cam_valid)[0]
            if zero_cam_ids.size > 0:
                zero_cams = [calib.camera_poses.names[i] for i in zero_cam_ids]
                summary = (
                    "\n" +
                    "=" * 70 + "\n" +
                    f"SUMMARY: Excluding {len(zero_cams)} camera(s) with no extrinsics\n" +
                    "=" * 70 + "\n" +
                    f"Cameras: {', '.join(zero_cams)}\n" +
                    "They will not appear in camera_poses export.\n" +
                    "=" * 70
                )
                info(summary)
        except Exception:
            pass

        # --- New View Rejection Step ---
        if reject_view_threshold is not None and reject_view_threshold > 0:
            info(f"Applying view rejection with threshold: {reject_view_threshold} px")
            errors, valid = tables.reprojection_error(
                calib.reprojected, calib.point_table
            )

            # Start with the mask from adjust_outliers
            view_mask = calib.inliers.copy()
            rejected_views_count = 0

            # Iterate through all camera/frame views
            # Use calib.size which should have cameras, rig_poses dimensions
            for cam_idx in range(calib.size.cameras):
                for frame_idx in range(calib.size.rig_poses):
                    # Check points valid *both* intrinsically and for this view
                    view_valid_mask = valid[cam_idx, frame_idx]
                    if not np.any(
                        view_valid_mask
                    ):  # Skip if no valid points in this view
                        continue

                    view_errors = errors[cam_idx, frame_idx][view_valid_mask]

                    # If any valid point in this view exceeds the threshold
                    if np.any(view_errors > reject_view_threshold):
                        if np.any(
                            view_mask[cam_idx, frame_idx]
                        ):  # Check if not already fully masked
                            rejected_views_count += 1
                        # Reject the entire view by setting its mask slice to False
                        view_mask[cam_idx, frame_idx] = False

            num_inliers_before = np.sum(calib.inliers)
            num_inliers_after = np.sum(view_mask)
            info(
                f"Rejected {rejected_views_count} views containing points with error > {reject_view_threshold} px."
            )
            info(
                f"Point count changed from {num_inliers_before} to {num_inliers_after}."
            )

            if rejected_views_count > 0:
                # Update calibration object with the new view mask
                calib = calib.copy(inlier_mask=view_mask)
                # Run bundle adjustment one last time with the view-based mask
                info("Running final bundle adjustment pass after view rejection.")
                # Determine f_scale for the final pass (e.g., using auto_scale logic or just 1.0)
                final_f_scale = (
                    select_threshold(quantile=quantile, factor=auto_scale)(
                        calib.reprojection_inliers
                    )
                    if auto_scale is not None
                    else 1.0
                )
                calib = calib.bundle_adjust(
                    loss=loss, tolerance=tolerance, f_scale=final_f_scale
                )
                calib.report(f"After final adjustment post view rejection for '{name}'")
            else:
                info("No additional views rejected based on threshold.")

        self.calibrations[name] = calib
        return calib

    @property
    def sizes(self):
        return self.names._map(len)

    @property
    def initialisation(self) -> Calibration:
        return self.calibrations["initialisation"]

    @property
    def latest_calibration(self) -> Calibration:
        return list(self.calibrations.values())[-1]

    @property
    def log_entries(self):
        return self.log_handler.records

    def has_calibrations(self):
        return len(self.calibrations) > 0

    def get_calibrations(self):
        return self.calibrations

    def push_calibration(self, name, calib):
        if name in self.calibrations:
            raise KeyError(
                f"calibration {name} exists already {list(self.calibrations.keys())}"
            )
        self.calibrations[name] = calib

    def get_camera_sets(self):
        if self.has_calibrations():
            return {k: calib.cameras for k, calib in self.calibrations.items()}

        if self.cameras is not None:
            return dict(initialisation=self.cameras)

    def export_json(self, master=None):
        master = master or self.names.camera[0]
        assert (
            master is None or master in self.names.camera
        ), f"master f{master} not found in cameras f{str(self.names.camera)}"

        calib = self.latest_calibration
        if master is not None:
            calib = calib.with_master(master)

        return export_json(calib, self.names, self.filenames, master=master)

    def export(self, filename=None, master=None):
        filename = filename or path.join(self.output_path, f"{self.name}.json")
        info(f"Exporting calibration to {filename}")

        data = self.export_json(master=master)
        with open(filename, "w") as outfile:
            json.dump(to_dicts(data), outfile, indent=2)

        # After exporting, clearly state which cameras were excluded from poses
        try:
            calib = self.latest_calibration if hasattr(self, 'calibrations') and self.has_calibrations() else None
            if calib is not None:
                cam_valid = calib.camera_poses.valid
                zero_cam_ids = np.where(~cam_valid)[0]
                if zero_cam_ids.size > 0:
                    zero_cams = [calib.camera_poses.names[i] for i in zero_cam_ids]
                    end_msg = (
                        "\n" +
                        "=" * 70 + "\n" +
                        f"EXPORT NOTICE: {len(zero_cams)} camera(s) excluded from camera_poses\n" +
                        "=" * 70 + "\n" +
                        f"Cameras: {', '.join(zero_cams)}\n" +
                        "Recommendation: Drop these cameras or recapture with boards.\n" +
                        "They remain in the 'cameras' intrinsics section but have no extrinsics.\n" +
                        "=" * 70
                    )
                    info(end_msg)
        except Exception:
            pass

    def dump(self, filename=None):
        filename = filename or path.join(self.output_path, f"{self.name}.pkl")

        info(f"Dumping state and history to {filename}")
        # Build a filtered copy that drops cameras without valid extrinsics
        ws_to_dump = self
        try:
            if self.has_calibrations():
                latest = self.latest_calibration
                cam_valid = np.array(latest.camera_poses.valid, dtype=bool)
                keep_idx = np.where(cam_valid)[0]

                if keep_idx.size < latest.size.cameras:
                    dropped = [latest.camera_poses.names[i] for i in np.where(~cam_valid)[0]]
                    info(
                        "\n" +
                        "=" * 70 + "\n" +
                        f"PKL CLEANUP: dropping {len(dropped)} camera(s) without extrinsics from dump\n" +
                        "=" * 70 + "\n" +
                        f"Cameras: {', '.join(dropped)}\n" +
                        "These cameras lacked valid extrinsic poses (0 pose count).\n" +
                        "=" * 70
                    )

                    # Helper to index python lists
                    def idx_list(xs, idx):
                        return [xs[i] for i in idx]

                    # Build a shallow filtered workspace instance
                    ws_filtered = Workspace(self.output_path, self.name)

                    # Names
                    filtered_cam_names = idx_list(self.names.camera, keep_idx)
                    ws_filtered.names = self.names._extend(camera=filtered_cam_names)

                    # Filenames and sizes if available
                    ws_filtered.filenames = idx_list(self.filenames, keep_idx) if self.filenames is not None else None
                    ws_filtered.image_path = self.image_path
                    ws_filtered.image_sizes = idx_list(self.image_sizes, keep_idx) if self.image_sizes is not None else None

                    # Boards/colors unchanged
                    ws_filtered.boards = self.boards
                    ws_filtered.board_colors = self.board_colors

                    # Cameras list filtered if available
                    ws_filtered.cameras = idx_list(self.cameras, keep_idx) if getattr(self, 'cameras', None) is not None else None

                    # Point table filtered along camera axis (0)
                    if self.point_table is not None:
                        ws_filtered.point_table = Table.create(
                            points=self.point_table.points[keep_idx, ...],
                            valid=self.point_table.valid[keep_idx, ...],
                        )
                    else:
                        ws_filtered.point_table = None

                    # Pose table (per-view estimates) filtered along camera axis (0)
                    if self.pose_table is not None:
                        kwargs = {}
                        # Slice known fields when present
                        if hasattr(self.pose_table, 'poses'):
                            kwargs['poses'] = self.pose_table.poses[keep_idx, ...]
                        if hasattr(self.pose_table, 'num_points'):
                            kwargs['num_points'] = self.pose_table.num_points[keep_idx, ...]
                        if hasattr(self.pose_table, 'valid'):
                            kwargs['valid'] = self.pose_table.valid[keep_idx, ...]
                        if hasattr(self.pose_table, 'reprojection_error'):
                            kwargs['reprojection_error'] = self.pose_table.reprojection_error[keep_idx, ...]
                        if hasattr(self.pose_table, 'view_angles'):
                            kwargs['view_angles'] = self.pose_table.view_angles[keep_idx, ...]
                        ws_filtered.pose_table = Table.create(**kwargs) if kwargs else None
                    else:
                        ws_filtered.pose_table = None

                    # Calibrations: rebuild each with filtered cameras and tables
                    ws_filtered.calibrations = OrderedDict()
                    for k, calib in self.calibrations.items():
                        # Cameras
                        filtered_cameras = idx_list(list(calib.cameras), keep_idx)
                        cameras_pl = ParamList(filtered_cameras, [self.names.camera[i] for i in keep_idx])

                        # Camera poses (per-camera)
                        cam_pose_tbl = calib.camera_poses.pose_table
                        cam_pose_tbl_f = Table.create(
                            poses=cam_pose_tbl.poses[keep_idx, ...],
                            valid=cam_pose_tbl.valid[keep_idx, ...],
                        )
                        camera_poses_ps = PoseSet(cam_pose_tbl_f, [calib.camera_poses.names[i] for i in keep_idx])

                        # Point table
                        pt = calib.point_table
                        pt_f = Table.create(points=pt.points[keep_idx, ...], valid=pt.valid[keep_idx, ...])

                        # Inlier mask (optional)
                        inliers_f = calib.inlier_mask[keep_idx, ...] if calib.inlier_mask is not None else None

                        # Rebuild calibration
                        calib_f = calib.copy(
                            cameras=cameras_pl,
                            camera_poses=camera_poses_ps,
                            point_table=pt_f,
                            inlier_mask=inliers_f,
                        )
                        ws_filtered.calibrations[k] = calib_f

                    # Keep logging, detections, etc.
                    ws_filtered.detections = None
                    ws_filtered.log_handler = self.log_handler

                    ws_to_dump = ws_filtered
        except Exception:
            # On any issue, fall back to dumping the original workspace
            pass

        with open(filename, "wb") as file:
            pickle.dump(ws_to_dump, file)

    @staticmethod
    def load(filename):
        assert path.isfile(filename), f"Workspace.load: file does not exist {filename}"
        with open(filename, "rb") as file:
            ws = pickle.load(file)
            return ws

    def fields(self, *keys):
        return subset(self.__dict__, keys)

    def __getstate__(self):
        return self.fields(
            "calibrations",
            "detections",
            "boards",
            "board_colors",
            "filenames",
            "image_path",
            "names",
            "image_sizes",
            "point_table",
            "pose_table",
            "log_handler",
        )

    def __setstate__(self, d):
        for k, v in d.items():
            self.__dict__[k] = v

        self.images = None
