"""
Distortion Diagnostic subcommand for multical.

Usage:
    multical diagnose --boards board.yaml --cameras Cam_001 Cam_002 --sample_images 30
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import os
import numpy as np
import cv2
from multiprocessing import cpu_count
from scipy import stats

from simple_parsing.helpers import list_field

from multical.config.runtime import find_board_config, find_camera_images
from multical.image.detect import common_image_size, load_images, detect_images
from multical.camera import calibration_points
from multical.io.logging import setup_logging, info
from structs.struct import split_dict, map_list

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def to_jsonable(obj):
    """Convert numpy/scalar containers to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

DISTORTION_MODELS = {
    'zero': {
        'flags': cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 0,
        'desc': 'No distortion'
    },
    'k1_only': {
        'flags': cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 1,
        'desc': 'k1 only'
    },
    'k1_k2': {
        'flags': cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 2,
        'desc': 'k1 + k2'
    },
    'standard': {
        'flags': cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 3,
        'desc': 'k1, k2, k3'
    },
    'full': {
        'flags': 0,
        'n_params': 5,
        'desc': 'k1, k2, k3, p1, p2'
    }
}


@dataclass
class Diagnose:
    """Diagnose optimal distortion model for cameras"""

    image_path: str = "."  # Path to calibration images
    boards: Optional[str] = None  # Board configuration file
    cameras: List[str] = list_field()  # Camera names
    camera_pattern: Optional[str] = None  # Camera path pattern
    sample_images: int = 30  # Images per camera to sample
    output_dir: str = "./distortion_diagnosis"  # Output directory
    num_threads: int = cpu_count() - 1  # Number of threads

    def execute(self):
        run_diagnosis(self)


def sample_by_coverage(detections, boards, image_size, n_samples, bins=8):
    """Sample images to maximize spatial coverage."""
    w, h = image_size
    x_bins = np.linspace(0, w, bins + 1)
    y_bins = np.linspace(0, h, bins + 1)

    coverage_scores = []
    for idx, frame_dets in enumerate(detections):
        corners = []
        for board, det in zip(boards, frame_dets):
            if board.has_min_detections(det):
                corners.append(det.corners)

        if not corners:
            coverage_scores.append((idx, 0, set()))
            continue

        pts = np.vstack(corners).reshape(-1, 2)
        xi = np.clip(np.digitize(pts[:, 0], x_bins) - 1, 0, bins - 1)
        yi = np.clip(np.digitize(pts[:, 1], y_bins) - 1, 0, bins - 1)
        covered = set(zip(xi, yi))
        coverage_scores.append((idx, len(covered), covered))

    # Greedy selection
    selected = []
    total_covered = set()
    while len(selected) < n_samples and coverage_scores:
        best_i = max(range(len(coverage_scores)),
                     key=lambda i: len(coverage_scores[i][2] - total_covered))
        idx, _, bins_covered = coverage_scores.pop(best_i)
        selected.append(idx)
        total_covered.update(bins_covered)

    return sorted(selected)


def calibrate_model(obj_pts, img_pts, image_size, model_name):
    """Run calibration with specified model."""
    flags = DISTORTION_MODELS[model_name]['flags']
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    try:
        rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
            obj_pts, img_pts, image_size, None, None, criteria=criteria, flags=flags
        )
        return {'success': True, 'rms': rms, 'K': K, 'dist': dist.flatten(),
                'rvecs': rvecs, 'tvecs': tvecs, 'n_pts': sum(len(p) for p in obj_pts)}
    except cv2.error as e:
        return {'success': False, 'error': str(e)}


def compute_spatial_errors(K, dist, rvecs, tvecs, obj_pts, img_pts, image_size):
    """Compute errors by image region (3x3 grid)."""
    w, h = image_size
    regions = {f'{y}_{x}': [] for y in ['top', 'mid', 'bot'] for x in ['left', 'center', 'right']}
    region_map = [['top_left', 'top_center', 'top_right'],
                  ['mid_left', 'mid_center', 'mid_right'],
                  ['bot_left', 'bot_center', 'bot_right']]

    x_bins = np.linspace(0, w, 4)
    y_bins = np.linspace(0, h, 4)

    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        img_flat = img.reshape(-1, 2)
        errors = np.linalg.norm(proj - img_flat, axis=1)

        xi = np.clip(np.digitize(img_flat[:, 0], x_bins) - 1, 0, 2)
        yi = np.clip(np.digitize(img_flat[:, 1], y_bins) - 1, 0, 2)

        for i, err in enumerate(errors):
            regions[region_map[yi[i]][xi[i]]].append(err)

    return {k: np.mean(v) if v else np.nan for k, v in regions.items()}


def compute_bic(rms, n_points, n_dist_params):
    """BIC = n*ln(MSE) + k*ln(n)"""
    k = 4 + n_dist_params  # 4 intrinsic + distortion
    mse = max(rms ** 2, 1e-10)
    return n_points * np.log(mse) + k * np.log(n_points)


def compute_coverage(img_pts: List[np.ndarray], image_size: Tuple[int, int], grid: int = 3) -> Dict[str, Any]:
    """Compute detection coverage per grid region.

    Args:
        img_pts: List of detected image points per frame.
        image_size: Image dimensions (width, height).
        grid: Grid size (default 3x3).

    Returns:
        Coverage statistics including center, edges, corners percentages.
    """
    w, h = image_size
    x_bins = np.linspace(0, w, grid + 1)
    y_bins = np.linspace(0, h, grid + 1)

    all_pts = np.vstack([p.reshape(-1, 2) for p in img_pts])
    xi = np.clip(np.digitize(all_pts[:, 0], x_bins) - 1, 0, grid - 1)
    yi = np.clip(np.digitize(all_pts[:, 1], y_bins) - 1, 0, grid - 1)

    counts = np.zeros((grid, grid))
    for x, y in zip(xi, yi):
        counts[int(y), int(x)] += 1

    total = counts.sum()
    if total == 0:
        return {'center': 0, 'edges': 0, 'corners': 0, 'regions_with_detections': 0, 'counts': counts.tolist()}

    center = counts[1, 1] / total
    edges = (counts[0, 1] + counts[2, 1] + counts[1, 0] + counts[1, 2]) / total
    corners = (counts[0, 0] + counts[0, 2] + counts[2, 0] + counts[2, 2]) / total

    return {
        'center': float(center),
        'edges': float(edges),
        'corners': float(corners),
        'regions_with_detections': int(np.sum(counts > 0)),
        'counts': counts.tolist()
    }


def compute_radial_profile(K: np.ndarray, dist: np.ndarray, rvecs: List[np.ndarray],
                          tvecs: List[np.ndarray], obj_pts: List[np.ndarray],
                          img_pts: List[np.ndarray]) -> Dict[str, Any]:
    """Compute error vs radius from principal point.

    Args:
        K: Camera intrinsic matrix.
        dist: Distortion coefficients.
        rvecs, tvecs: Rotation and translation vectors per frame.
        obj_pts, img_pts: Object and image points per frame.

    Returns:
        Radial profile with correlation, slope, and interpretation.
    """
    cx, cy = K[0, 2], K[1, 2]
    radii, errors = [], []

    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        img_flat = img.reshape(-1, 2)

        r = np.sqrt((img_flat[:, 0] - cx)**2 + (img_flat[:, 1] - cy)**2)
        err = np.linalg.norm(proj - img_flat, axis=1)
        radii.extend(r)
        errors.extend(err)

    radii, errors = np.array(radii), np.array(errors)

    if len(radii) < 2:
        return {'correlation': 0, 'slope': 0, 'interpretation': 'insufficient_data',
                'radii': [], 'errors': []}

    correlation = float(np.corrcoef(radii, errors)[0, 1])
    slope, intercept = np.polyfit(radii, errors, 1)

    # Interpret slope: > 0.0005 px/px suggests distortion
    if slope > 0.001:
        interpretation = 'strong_radial_pattern'
    elif slope > 0.0005:
        interpretation = 'mild_radial_pattern'
    else:
        interpretation = 'no_radial_pattern'

    return {
        'correlation': correlation,
        'slope': float(slope),
        'intercept': float(intercept),
        'interpretation': interpretation,
        'radii': radii.tolist(),
        'errors': errors.tolist()
    }


def compute_residual_direction(K: np.ndarray, dist: np.ndarray, rvecs: List[np.ndarray],
                               tvecs: List[np.ndarray], obj_pts: List[np.ndarray],
                               img_pts: List[np.ndarray]) -> Dict[str, Any]:
    """Check if residuals point radially (distortion) or randomly (noise).

    Args:
        K: Camera intrinsic matrix.
        dist: Distortion coefficients.
        rvecs, tvecs: Rotation and translation vectors per frame.
        obj_pts, img_pts: Object and image points per frame.

    Returns:
        Radial alignment score and interpretation.
    """
    cx, cy = K[0, 2], K[1, 2]
    radial_alignments = []

    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        img_flat = img.reshape(-1, 2)

        # Error vector and radial direction
        error_vec = proj - img_flat
        radial_vec = img_flat - np.array([cx, cy])

        for ev, rv in zip(error_vec, radial_vec):
            err_mag = np.linalg.norm(ev)
            rad_mag = np.linalg.norm(rv)
            # Only consider points with measurable error and not at center
            if err_mag > 0.1 and rad_mag > 10:
                cos_angle = np.dot(ev, rv) / (err_mag * rad_mag)
                radial_alignments.append(abs(cos_angle))

    if not radial_alignments:
        return {'radial_alignment': 0, 'interpretation': 'insufficient_data', 'n_points': 0}

    mean_alignment = float(np.mean(radial_alignments))

    # Interpret: random noise would give ~0.5 (uniform distribution of angles)
    # Radial distortion would give values closer to 1.0
    if mean_alignment > 0.7:
        interpretation = 'strong_radial_pattern'
    elif mean_alignment > 0.55:
        interpretation = 'mild_radial_pattern'
    else:
        interpretation = 'random_noise'

    return {
        'radial_alignment': mean_alignment,
        'interpretation': interpretation,
        'n_points': len(radial_alignments)
    }


def f_test_models(result_simple: Dict, result_complex: Dict, n_params_simple: int,
                  n_params_complex: int) -> Dict[str, Any]:
    """F-test comparing nested distortion models.

    Args:
        result_simple: Calibration result from simpler model.
        result_complex: Calibration result from more complex model.
        n_params_simple: Number of distortion parameters in simple model.
        n_params_complex: Number of distortion parameters in complex model.

    Returns:
        F-statistic, p-value, and significance flag.
    """
    if not result_simple.get('success') or not result_complex.get('success'):
        return {'f_statistic': None, 'p_value': None, 'significant': False}

    # RSS = RMS^2 * n_points (sum of squared residuals)
    rss_simple = result_simple['rms']**2 * result_simple['n_pts']
    rss_complex = result_complex['rms']**2 * result_complex['n_pts']

    n_pts = result_complex['n_pts']
    # Total parameters: 4 intrinsic (fx, fy, cx, cy) + distortion + 6 per view
    df_diff = n_params_complex - n_params_simple
    df_full = n_pts - (4 + n_params_complex)

    if df_full <= 0 or df_diff <= 0 or rss_complex <= 0:
        return {'f_statistic': None, 'p_value': None, 'significant': False}

    f_stat = ((rss_simple - rss_complex) / df_diff) / (rss_complex / df_full)

    if f_stat < 0:
        # Complex model is worse (shouldn't happen with nested models)
        return {'f_statistic': float(f_stat), 'p_value': 1.0, 'significant': False}

    p_value = 1 - stats.f.cdf(f_stat, df_diff, df_full)

    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'df_numerator': df_diff,
        'df_denominator': df_full
    }


def compute_coefficient_consistency(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Check if distortion coefficients are consistent across cameras.

    Args:
        all_results: Dictionary of per-camera calibration results.

    Returns:
        Statistics on k1, k2 consistency across cameras.
    """
    k1_values = []
    k2_values = []

    for cam, res in all_results.items():
        full_result = res.get('models', {}).get('full', {})
        if full_result.get('success') and full_result.get('dist') is not None:
            dist = full_result['dist']
            if isinstance(dist, np.ndarray):
                dist = dist.tolist()
            k1_values.append(dist[0])
            k2_values.append(dist[1])

    if len(k1_values) < 2:
        return {
            'k1_mean': None, 'k1_std': None, 'k1_cv': None,
            'k2_mean': None, 'k2_std': None, 'k2_cv': None,
            'interpretation': 'insufficient_cameras'
        }

    k1_mean, k1_std = float(np.mean(k1_values)), float(np.std(k1_values))
    k2_mean, k2_std = float(np.mean(k2_values)), float(np.std(k2_values))

    # Coefficient of variation (handle near-zero means)
    k1_cv = abs(k1_std / k1_mean) if abs(k1_mean) > 1e-6 else float('inf')
    k2_cv = abs(k2_std / k2_mean) if abs(k2_mean) > 1e-6 else float('inf')

    # Interpret consistency
    if k1_cv < 0.5:
        interpretation = 'consistent'
    elif k1_cv < 1.0:
        interpretation = 'moderate'
    else:
        interpretation = 'inconsistent'

    return {
        'k1_mean': k1_mean,
        'k1_std': k1_std,
        'k1_cv': float(k1_cv) if not np.isinf(k1_cv) else None,
        'k2_mean': k2_mean,
        'k2_std': k2_std,
        'k2_cv': float(k2_cv) if not np.isinf(k2_cv) else None,
        'n_cameras': len(k1_values),
        'interpretation': interpretation
    }


def run_diagnosis(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("INFO", [])

    info(f"Distortion Diagnosis")
    info(f"{'='*60}")
    info(f"Image path: {args.image_path}")
    info(f"Sample images: {args.sample_images}")

    # Load boards
    boards_dict = find_board_config(args.image_path, board_file=args.boards)
    board_names, boards = split_dict(boards_dict)
    info(f"Boards: {board_names}")

    # Find cameras
    camera_images = find_camera_images(
        args.image_path,
        args.cameras if args.cameras else None,
        args.camera_pattern,
        limit=None
    )
    camera_names = camera_images.cameras
    info(f"Cameras: {camera_names}")

    # Load images
    info("Loading images...")
    images = load_images(camera_images.filenames, j=args.num_threads,
                        prefix=camera_images.image_path)
    image_sizes = map_list(common_image_size, images)

    # Detect boards
    info("Detecting boards...")
    detected_points = detect_images(boards, images, j=args.num_threads)

    # Diagnose each camera
    all_results = {}

    for cam_idx, cam_name in enumerate(camera_names):
        info(f"\n{'='*60}")
        info(f"Diagnosing {cam_name}")

        cam_dets = detected_points[cam_idx]
        cam_size = image_sizes[cam_idx]

        # Sample images
        selected = sample_by_coverage(cam_dets, boards, cam_size, args.sample_images)
        info(f"  Selected {len(selected)} images")

        # Prepare calibration data
        sel_dets = [cam_dets[i] for i in selected]
        points = calibration_points(boards, sel_dets)

        if len(points.corners) < 10:
            info(f"  WARNING: Only {len(points.corners)} valid frames, skipping")
            continue

        obj_pts = [np.array(p, dtype=np.float32) for p in points.object_points]
        img_pts = [np.array(p, dtype=np.float32) for p in points.corners]

        # Compute coverage analysis
        coverage = compute_coverage(img_pts, cam_size)
        info(f"  Coverage: center={coverage['center']:.1%}, edges={coverage['edges']:.1%}, corners={coverage['corners']:.1%}")

        # Test each model
        cam_results = {'models': {}, 'coverage': coverage}
        for model_name, config in DISTORTION_MODELS.items():
            result = calibrate_model(obj_pts, img_pts, cam_size, model_name)

            if result['success']:
                result['bic'] = compute_bic(result['rms'], result['n_pts'], config['n_params'])
                result['spatial'] = compute_spatial_errors(
                    result['K'], result['dist'], result['rvecs'], result['tvecs'],
                    obj_pts, img_pts, cam_size
                )
                info(f"  {model_name:12s}: RMS={result['rms']:.3f}  BIC={result['bic']:.0f}")
            else:
                info(f"  {model_name:12s}: FAILED")

            cam_results['models'][model_name] = result

        # Find best model
        bics = {m: r['bic'] for m, r in cam_results['models'].items() if r.get('success')}
        if bics:
            best = min(bics, key=bics.get)
            cam_results['best_model'] = best
            info(f"  BEST: {best}")

        # Compute radial error profile using zero-distortion model
        zero_result = cam_results['models'].get('zero', {})
        if zero_result.get('success'):
            radial_profile = compute_radial_profile(
                zero_result['K'], zero_result['dist'],
                zero_result['rvecs'], zero_result['tvecs'],
                obj_pts, img_pts
            )
            cam_results['radial_profile'] = {
                'correlation': radial_profile['correlation'],
                'slope': radial_profile['slope'],
                'interpretation': radial_profile['interpretation']
            }
            info(f"  Radial profile: corr={radial_profile['correlation']:.3f}, slope={radial_profile['slope']:.5f} ({radial_profile['interpretation']})")

            # Compute residual direction analysis
            residual_dir = compute_residual_direction(
                zero_result['K'], zero_result['dist'],
                zero_result['rvecs'], zero_result['tvecs'],
                obj_pts, img_pts
            )
            cam_results['residual_direction'] = residual_dir
            info(f"  Residual direction: alignment={residual_dir['radial_alignment']:.2f} ({residual_dir['interpretation']})")

        # F-tests between nested models
        f_tests = {}
        model_pairs = [
            ('zero', 'k1_only', 0, 1),
            ('k1_only', 'k1_k2', 1, 2),
            ('k1_k2', 'standard', 2, 3),
            ('standard', 'full', 3, 5),
            ('zero', 'full', 0, 5),
        ]
        for simple, complex_, n_simple, n_complex in model_pairs:
            simple_res = cam_results['models'].get(simple, {})
            complex_res = cam_results['models'].get(complex_, {})
            f_result = f_test_models(simple_res, complex_res, n_simple, n_complex)
            f_tests[f'{simple}_vs_{complex_}'] = f_result
            if f_result.get('p_value') is not None:
                sig = '*' if f_result['significant'] else ''
                info(f"  F-test {simple} vs {complex_}: p={f_result['p_value']:.4f}{sig}")

        cam_results['f_tests'] = f_tests

        all_results[cam_name] = cam_results

    # Summary
    info(f"\n{'='*60}")
    info("SUMMARY")
    info(f"{'='*60}")

    best_models = [r.get('best_model', 'unknown') for r in all_results.values()]
    from collections import Counter
    model_counts = Counter(best_models)
    info(f"Best model votes: {dict(model_counts)}")

    # Cross-camera coefficient consistency
    consistency = compute_coefficient_consistency(all_results)
    info(f"\nCross-camera consistency:")
    if consistency['k1_mean'] is not None:
        k1_cv = consistency.get('k1_cv')
        k1_cv_str = f"{k1_cv:.2f}" if k1_cv is not None else "N/A"
        info(f"  k1: mean={consistency['k1_mean']:.4f}, std={consistency['k1_std']:.4f}, CV={k1_cv_str}")
        info(f"  k2: mean={consistency['k2_mean']:.4f}, std={consistency['k2_std']:.4f}")
        info(f"  Interpretation: {consistency['interpretation']}")
    else:
        info(f"  {consistency['interpretation']}")

    # Check spatial pattern (edge vs center)
    edge_ratios = []
    for cam, res in all_results.items():
        zero_spatial = res['models'].get('zero', {}).get('spatial', {})
        if zero_spatial:
            center = zero_spatial.get('mid_center', 1)
            edges = [zero_spatial.get(k, center) for k in
                    ['top_center', 'bot_center', 'mid_left', 'mid_right']]
            if center > 0:
                edge_ratios.append(np.nanmean(edges) / center)

    if edge_ratios:
        avg_ratio = np.mean(edge_ratios)
        info(f"\nEdge/Center error ratio (zero model): {avg_ratio:.2f}")
        if avg_ratio > 1.5:
            info("  -> Edges have higher error, suggesting REAL DISTORTION exists")

    # Aggregate radial profile results
    radial_interpretations = [r.get('radial_profile', {}).get('interpretation', 'unknown')
                              for r in all_results.values()]
    radial_counts = Counter(radial_interpretations)
    info(f"\nRadial profile interpretations: {dict(radial_counts)}")

    # Aggregate residual direction results
    residual_interpretations = [r.get('residual_direction', {}).get('interpretation', 'unknown')
                                for r in all_results.values()]
    residual_counts = Counter(residual_interpretations)
    info(f"Residual direction interpretations: {dict(residual_counts)}")

    # Count significant F-tests (zero vs full)
    n_significant_f = sum(1 for r in all_results.values()
                         if r.get('f_tests', {}).get('zero_vs_full', {}).get('significant', False))
    info(f"\nF-test (zero vs full) significant: {n_significant_f}/{len(all_results)} cameras")

    # Aggregate coverage warnings
    low_edge_coverage = [cam for cam, res in all_results.items()
                         if res.get('coverage', {}).get('edges', 0) < 0.1]
    if low_edge_coverage:
        info(f"\nWARNING: Low edge coverage (<10%) for: {', '.join(low_edge_coverage)}")
        info("  -> Edge distortion analysis may be unreliable for these cameras")

    # Improved recommendation using multiple metrics
    most_common = model_counts.most_common(1)[0][0] if model_counts else 'k1_k2'

    # Confidence assessment
    evidence_for_distortion = 0
    evidence_against_distortion = 0

    # BIC-based evidence
    if most_common in ['full', 'standard', 'k1_k2']:
        evidence_for_distortion += 1
    elif most_common == 'zero':
        evidence_against_distortion += 1

    # F-test evidence
    if n_significant_f > len(all_results) * 0.5:
        evidence_for_distortion += 2  # Strong evidence
    elif n_significant_f < len(all_results) * 0.2:
        evidence_against_distortion += 1

    # Radial profile evidence
    n_radial_pattern = radial_counts.get('mild_radial_pattern', 0) + radial_counts.get('strong_radial_pattern', 0)
    if n_radial_pattern > len(all_results) * 0.5:
        evidence_for_distortion += 1
    elif radial_counts.get('no_radial_pattern', 0) > len(all_results) * 0.7:
        evidence_against_distortion += 1

    # Coefficient consistency evidence
    if consistency['interpretation'] == 'consistent' and consistency.get('k1_mean'):
        if abs(consistency['k1_mean']) > 0.005:
            evidence_for_distortion += 1  # Consistent non-zero k1

    # Determine confidence
    if evidence_for_distortion >= 3:
        confidence = 'high'
    elif evidence_for_distortion >= 2 or evidence_against_distortion == 0:
        confidence = 'medium'
    else:
        confidence = 'low'

    info(f"\n{'='*60}")
    info("RECOMMENDATION")
    info(f"{'='*60}")
    info(f"Evidence score: +{evidence_for_distortion} distortion, +{evidence_against_distortion} none")
    info(f"Confidence: {confidence}")

    if evidence_for_distortion > evidence_against_distortion:
        if most_common in ['full']:
            info("\nYour cameras have significant distortion including tangential.")
            info("Use: --fix_radial False --fix_tangential False")
            final_recommendation = 'full'
        else:
            info("\nYour cameras have radial distortion (k1, k2).")
            info("Use: --fix_radial False --fix_tangential True")
            final_recommendation = 'radial'
    else:
        info("\nYour cameras have minimal distortion.")
        info("Use: --fix_radial True --fix_tangential True")
        final_recommendation = 'zero'

    # Save report with all metrics
    report = {
        'cameras': {cam: {
            'best_model': res.get('best_model'),
            'models': {m: {
                'rms': r.get('rms'),
                'bic': r.get('bic'),
                'dist': (r.get('dist').tolist() if isinstance(r.get('dist'), np.ndarray)
                        else r.get('dist')) if r.get('success') else None
            } for m, r in res['models'].items()},
            'coverage': res.get('coverage'),
            'radial_profile': res.get('radial_profile'),
            'residual_direction': res.get('residual_direction'),
            'f_tests': {k: {kk: vv for kk, vv in v.items() if kk != 'df_numerator' and kk != 'df_denominator'}
                       for k, v in res.get('f_tests', {}).items()}
        } for cam, res in all_results.items()},
        'cross_camera': {
            'consistency': consistency,
            'bic_model_votes': dict(model_counts),
            'radial_pattern_votes': dict(radial_counts),
            'residual_direction_votes': dict(residual_counts),
            'f_test_significant_count': n_significant_f,
            'f_test_total_cameras': len(all_results)
        },
        'recommendation': final_recommendation,
        'bic_recommendation': most_common,
        'confidence': confidence,
        'evidence': {
            'for_distortion': evidence_for_distortion,
            'against_distortion': evidence_against_distortion
        },
        'edge_center_ratio': float(np.mean(edge_ratios)) if edge_ratios else None
    }

    report_path = output_dir / 'distortion_diagnosis.json'
    with open(report_path, 'w') as f:
        json.dump(to_jsonable(report), f, indent=2)

    info(f"\nReport saved to: {report_path}")

    # Create plots
    if all_results:
        cams = list(all_results.keys())
        n_cams = len(cams)

        # Figure 1: BIC Model Comparison
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(n_cams)
        width = 0.15

        for i, model in enumerate(DISTORTION_MODELS.keys()):
            bics = [all_results[c]['models'].get(model, {}).get('bic', np.nan) for c in cams]
            ax1.bar(x + i * width, bics, width, label=model)

        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(cams, rotation=45, ha='right')
        ax1.set_ylabel('BIC Score (lower = better)')
        ax1.set_title('Distortion Model Comparison by BIC')
        ax1.legend(loc='upper right')
        plt.tight_layout()

        for ext in ['png', 'svg']:
            fig1.savefig(output_dir / f'model_comparison.{ext}')

        # Figure 2: Coverage Analysis
        fig2, axes2 = plt.subplots(1, min(n_cams, 6), figsize=(3 * min(n_cams, 6), 3))
        if n_cams == 1:
            axes2 = [axes2]
        for idx, (cam, ax) in enumerate(zip(cams[:6], axes2)):
            coverage = all_results[cam].get('coverage', {})
            counts = coverage.get('counts', [[0]*3]*3)
            im = ax.imshow(counts, cmap='YlOrRd', aspect='equal')
            ax.set_title(cam, fontsize=10)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['L', 'C', 'R'])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['T', 'M', 'B'])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{int(counts[i][j])}', ha='center', va='center', fontsize=8)

        fig2.suptitle('Detection Coverage (point counts per region)', fontsize=12)
        plt.tight_layout()

        for ext in ['png', 'svg']:
            fig2.savefig(output_dir / f'coverage_analysis.{ext}')

        # Figure 3: Radial Profile Summary
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        slopes = [all_results[c].get('radial_profile', {}).get('slope', 0) for c in cams]
        correlations = [all_results[c].get('radial_profile', {}).get('correlation', 0) for c in cams]

        x = np.arange(n_cams)
        width = 0.35
        ax3.bar(x - width/2, slopes, width, label='Slope (px/px)', color='steelblue')
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, correlations, width, label='Correlation', color='coral', alpha=0.7)

        ax3.axhline(y=0.0005, color='steelblue', linestyle='--', alpha=0.5, label='Mild threshold')
        ax3.axhline(y=0.001, color='steelblue', linestyle=':', alpha=0.5, label='Strong threshold')

        ax3.set_xticks(x)
        ax3.set_xticklabels(cams, rotation=45, ha='right')
        ax3.set_ylabel('Slope (error increase per pixel radius)', color='steelblue')
        ax3_twin.set_ylabel('Correlation', color='coral')
        ax3.set_title('Radial Error Profile (zero-distortion model)')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        plt.tight_layout()

        for ext in ['png', 'svg']:
            fig3.savefig(output_dir / f'radial_profile.{ext}')

        # Figure 4: F-test Results
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        f_test_key = 'zero_vs_full'
        p_values = [all_results[c].get('f_tests', {}).get(f_test_key, {}).get('p_value', 1.0) for c in cams]

        colors = ['green' if p < 0.05 else 'gray' for p in p_values]
        bars = ax4.bar(cams, [-np.log10(max(p, 1e-10)) for p in p_values], color=colors)

        ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 threshold')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title(f'F-test: Zero vs Full Model (green = significant)')
        ax4.set_xticks(np.arange(len(cams)))
        ax4.set_xticklabels(cams, rotation=45, ha='right')
        ax4.legend()
        plt.tight_layout()

        for ext in ['png', 'svg']:
            fig4.savefig(output_dir / f'f_test_results.{ext}')

        # Figure 5: k1 Coefficient Distribution
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        k1_values = [all_results[c]['models'].get('full', {}).get('dist', [0])[0]
                     if all_results[c]['models'].get('full', {}).get('success') else 0
                     for c in cams]

        ax5.bar(cams, k1_values, color='steelblue')
        if consistency.get('k1_mean') is not None:
            ax5.axhline(y=consistency['k1_mean'], color='red', linestyle='-', label=f"Mean: {consistency['k1_mean']:.4f}")
            ax5.axhline(y=consistency['k1_mean'] + consistency['k1_std'], color='red', linestyle='--', alpha=0.5)
            ax5.axhline(y=consistency['k1_mean'] - consistency['k1_std'], color='red', linestyle='--', alpha=0.5, label=f"±Std: {consistency['k1_std']:.4f}")

        ax5.set_ylabel('k1 Coefficient')
        k1_cv_val = consistency.get('k1_cv')
        k1_cv_str = f"{k1_cv_val:.2f}" if k1_cv_val is not None else "N/A"
        ax5.set_title(f"k1 Distribution Across Cameras (CV={k1_cv_str})")
        ax5.set_xticks(np.arange(len(cams)))
        ax5.set_xticklabels(cams, rotation=45, ha='right')
        ax5.legend()
        plt.tight_layout()

        for ext in ['png', 'svg']:
            fig5.savefig(output_dir / f'k1_distribution.{ext}')

        info(f"\nPlots saved to: {output_dir}")
        plt.show()
