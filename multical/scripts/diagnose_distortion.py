"""
Distortion Diagnostic Tool for Multi-Camera Calibration

Quickly analyzes calibration images to determine optimal distortion model.
Compares multiple models (zero, k1-only, k1+k2, full standard) using BIC
and spatial reprojection error analysis.

Usage:
    python -m multical.scripts.diagnose_distortion \
        --image_path /path/to/calibration \
        --boards board_config.yaml \
        --cameras Cam_001 Cam_002 ... \
        --sample_images 30 \
        --output_dir ./distortion_diagnosis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
import cv2
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

from simple_parsing import ArgumentParser

import matplotlib
matplotlib.use('TkAgg')  # For interactive windows
import matplotlib.pyplot as plt

# Multical imports
from multical.config import find_board_config, find_camera_images
from multical.image.detect import detect_images, load_images
from multical.camera import calibration_points, board_correspondences
from structs.struct import transpose_lists


# Apply Nature journal style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


@dataclass
class DiagnoseArgs:
    """Distortion diagnostic tool arguments"""

    image_path: str = "."  # Path to calibration images
    boards: str = "boards.yaml"  # Board configuration file
    cameras: List[str] = field(default_factory=list)  # Camera names
    camera_pattern: Optional[str] = None  # Camera path pattern
    sample_images: int = 30  # Number of images to sample per camera
    output_dir: str = "./distortion_diagnosis"  # Output directory
    num_threads: int = cpu_count() - 1  # Number of threads


# Distortion model configurations
DISTORTION_MODELS = {
    'zero': {
        'flags': cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 0,
        'description': 'No distortion (k1=k2=k3=p1=p2=0)'
    },
    'k1_only': {
        'flags': cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 1,
        'description': 'Radial k1 only'
    },
    'k1_k2': {
        'flags': cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 2,
        'description': 'Radial k1 + k2'
    },
    'standard': {
        'flags': cv2.CALIB_ZERO_TANGENT_DIST,
        'n_params': 3,
        'description': 'Full radial (k1, k2, k3), no tangential'
    },
    'full': {
        'flags': 0,
        'n_params': 5,
        'description': 'Full standard (k1, k2, k3, p1, p2)'
    }
}


def sample_images_by_coverage(
    detections: List,
    boards: List,
    image_size: Tuple[int, int],
    n_samples: int,
    approx_bins: int = 8
) -> List[int]:
    """
    Sample images to maximize spatial coverage across the image.

    Args:
        detections: List of board detections per image
        boards: List of board objects
        image_size: (width, height) of images
        n_samples: Number of images to select
        approx_bins: Grid size for coverage calculation

    Returns:
        List of selected image indices
    """
    # Create bins for coverage
    w, h = image_size
    x_bins = np.linspace(0, w, approx_bins + 1)
    y_bins = np.linspace(0, h, approx_bins + 1)

    # Calculate coverage for each image
    coverage_scores = []
    for img_idx, frame_detections in enumerate(detections):
        all_corners = []
        for board, det in zip(boards, frame_detections):
            if board.has_min_detections(det):
                all_corners.append(det.corners)

        if not all_corners:
            coverage_scores.append((img_idx, 0, set()))
            continue

        corners = np.vstack(all_corners).reshape(-1, 2)

        # Compute which bins are covered
        x_idx = np.digitize(corners[:, 0], x_bins) - 1
        y_idx = np.digitize(corners[:, 1], y_bins) - 1
        x_idx = np.clip(x_idx, 0, approx_bins - 1)
        y_idx = np.clip(y_idx, 0, approx_bins - 1)

        covered_bins = set(zip(x_idx, y_idx))
        coverage_scores.append((img_idx, len(covered_bins), covered_bins))

    # Greedy selection to maximize total coverage
    selected = []
    total_covered = set()

    while len(selected) < n_samples and coverage_scores:
        # Find image that adds most new coverage
        best_idx = -1
        best_new_coverage = -1
        best_entry = None

        for i, (img_idx, _, bins) in enumerate(coverage_scores):
            new_coverage = len(bins - total_covered)
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_idx = i
                best_entry = coverage_scores[i]

        if best_idx >= 0:
            img_idx, _, bins = best_entry
            selected.append(img_idx)
            total_covered.update(bins)
            coverage_scores.pop(best_idx)
        else:
            break

    return sorted(selected)


def calibrate_with_model(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int],
    model_name: str,
    max_iter: int = 30
) -> Dict:
    """
    Run OpenCV calibration with specified distortion model.

    Returns:
        Dictionary with calibration results
    """
    model_config = DISTORTION_MODELS[model_name]
    flags = model_config['flags']

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1e-6)

    try:
        rms, K, dist, rvecs, tvecs, _, _, per_view_errors = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            image_size,
            None,
            None,
            criteria=criteria,
            flags=flags
        )

        return {
            'success': True,
            'rms': rms,
            'K': K,
            'dist': dist.flatten(),
            'rvecs': rvecs,
            'tvecs': tvecs,
            'per_view_errors': per_view_errors.flatten(),
            'n_images': len(object_points),
            'n_points': sum(len(pts) for pts in object_points)
        }
    except cv2.error as e:
        return {
            'success': False,
            'error': str(e)
        }


def compute_spatial_errors(
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int],
    grid_size: int = 3
) -> Dict[str, float]:
    """
    Compute reprojection errors binned by spatial region.

    Args:
        K, dist, rvecs, tvecs: Calibration results
        object_points, image_points: Calibration data
        image_size: (width, height)
        grid_size: NxN grid for spatial binning

    Returns:
        Dictionary mapping region names to mean errors
    """
    w, h = image_size
    x_bins = np.linspace(0, w, grid_size + 1)
    y_bins = np.linspace(0, h, grid_size + 1)

    # Region names for 3x3 grid
    region_names = [
        ['top_left', 'top_center', 'top_right'],
        ['mid_left', 'center', 'mid_right'],
        ['bot_left', 'bot_center', 'bot_right']
    ]

    region_errors = {name: [] for row in region_names for name in row}

    for obj_pts, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        # Project points
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        img_pts_flat = img_pts.reshape(-1, 2)

        # Compute per-point errors
        errors = np.linalg.norm(projected - img_pts_flat, axis=1)

        # Bin by spatial location
        x_idx = np.digitize(img_pts_flat[:, 0], x_bins) - 1
        y_idx = np.digitize(img_pts_flat[:, 1], y_bins) - 1
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)

        for pt_idx in range(len(errors)):
            xi, yi = x_idx[pt_idx], y_idx[pt_idx]
            region_name = region_names[yi][xi]
            region_errors[region_name].append(errors[pt_idx])

    # Compute mean error per region
    spatial_stats = {}
    for region, errs in region_errors.items():
        if errs:
            spatial_stats[region] = {
                'mean': float(np.mean(errs)),
                'std': float(np.std(errs)),
                'n': len(errs)
            }
        else:
            spatial_stats[region] = {'mean': np.nan, 'std': np.nan, 'n': 0}

    return spatial_stats


def compute_bic(rms: float, n_points: int, n_dist_params: int) -> float:
    """
    Compute Bayesian Information Criterion for model selection.

    BIC = n * ln(MSE) + k * ln(n)

    where n = number of observations, k = number of parameters,
    MSE = mean squared error

    Lower BIC is better.
    """
    # Intrinsic params: fx, fy, cx, cy = 4
    # Plus distortion params
    k = 4 + n_dist_params
    mse = rms ** 2

    # Avoid log(0)
    if mse <= 0:
        mse = 1e-10

    bic = n_points * np.log(mse) + k * np.log(n_points)
    return bic


def classify_coefficient(value: float, name: str) -> str:
    """Classify distortion coefficient significance."""
    thresholds = {
        'k1': (0.01, 0.1),    # (marginal, significant)
        'k2': (0.001, 0.01),
        'k3': (0.0001, 0.001),
        'p1': (0.0001, 0.001),
        'p2': (0.0001, 0.001),
    }

    low, high = thresholds.get(name, (0.001, 0.01))
    abs_val = abs(value)

    if abs_val < low:
        return 'negligible'
    elif abs_val < high:
        return 'marginal'
    else:
        return 'significant'


def plot_spatial_heatmap(
    spatial_errors: Dict[str, Dict],
    camera_name: str,
    model_name: str,
    output_dir: Path
) -> None:
    """Create spatial error heatmap for a single model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract 3x3 grid values
    region_order = [
        ['top_left', 'top_center', 'top_right'],
        ['mid_left', 'center', 'mid_right'],
        ['bot_left', 'bot_center', 'bot_right']
    ]

    grid = np.zeros((3, 3))
    for i, row in enumerate(region_order):
        for j, region in enumerate(row):
            grid[i, j] = spatial_errors[region]['mean']

    im = ax.imshow(grid, cmap='RdYlGn_r', vmin=0)

    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = grid[i, j]
            text_color = 'white' if val > np.nanmax(grid) * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=text_color, fontsize=12, fontweight='bold')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Left', 'Center', 'Right'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Top', 'Middle', 'Bottom'])

    ax.set_title(f'{camera_name} - {model_name}\nSpatial Reprojection Error (px)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Error (px)')

    # Save
    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'spatial_error_{camera_name}_{model_name}.{ext}')

    plt.close(fig)


def plot_model_comparison(
    results: Dict[str, Dict],
    camera_name: str,
    output_dir: Path
) -> None:
    """Create bar chart comparing models by BIC and RMS."""
    models = list(results.keys())
    bics = [results[m]['bic'] for m in models]
    rms_vals = [results[m]['rms'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))

    # BIC comparison
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(x, bics, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('BIC Score')
    ax.set_title(f'{camera_name}\nModel Comparison (BIC - lower is better)')

    # Highlight best model
    best_idx = np.argmin(bics)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    # RMS comparison
    ax = axes[1]
    bars = ax.bar(x, rms_vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('RMS Error (px)')
    ax.set_title(f'{camera_name}\nReprojection Error')

    plt.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'model_comparison_{camera_name}.{ext}')

    plt.close(fig)


def plot_coefficient_summary(
    all_camera_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot distortion coefficients across all cameras."""
    cameras = list(all_camera_results.keys())

    # Extract coefficients from 'full' model
    k1_vals = []
    k2_vals = []
    k3_vals = []
    p1_vals = []
    p2_vals = []

    for cam in cameras:
        dist = all_camera_results[cam]['models']['full'].get('dist', np.zeros(5))
        if len(dist) >= 5:
            k1_vals.append(dist[0])
            k2_vals.append(dist[1])
            p1_vals.append(dist[2])
            p2_vals.append(dist[3])
            k3_vals.append(dist[4])
        else:
            k1_vals.append(0)
            k2_vals.append(0)
            p1_vals.append(0)
            p2_vals.append(0)
            k3_vals.append(0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = np.arange(len(cameras))

    # k1
    ax = axes[0, 0]
    ax.bar(x, k1_vals, color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=90)
    ax.set_ylabel('k1')
    ax.set_title('Radial Distortion k1')

    # k2
    ax = axes[0, 1]
    ax.bar(x, k2_vals, color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=90)
    ax.set_ylabel('k2')
    ax.set_title('Radial Distortion k2')

    # k3
    ax = axes[0, 2]
    ax.bar(x, k3_vals, color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=90)
    ax.set_ylabel('k3')
    ax.set_title('Radial Distortion k3')

    # p1
    ax = axes[1, 0]
    ax.bar(x, p1_vals, color='coral')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=90)
    ax.set_ylabel('p1')
    ax.set_title('Tangential Distortion p1')

    # p2
    ax = axes[1, 1]
    ax.bar(x, p2_vals, color='coral')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=90)
    ax.set_ylabel('p2')
    ax.set_title('Tangential Distortion p2')

    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = (
        f"Summary Statistics (Full Model)\n"
        f"{'='*40}\n\n"
        f"k1: mean={np.mean(k1_vals):.4f}, std={np.std(k1_vals):.4f}\n"
        f"k2: mean={np.mean(k2_vals):.6f}, std={np.std(k2_vals):.6f}\n"
        f"k3: mean={np.mean(k3_vals):.6f}, std={np.std(k3_vals):.6f}\n"
        f"p1: mean={np.mean(p1_vals):.6f}, std={np.std(p1_vals):.6f}\n"
        f"p2: mean={np.mean(p2_vals):.6f}, std={np.std(p2_vals):.6f}\n"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'coefficient_summary.{ext}')

    plt.close(fig)


def plot_bic_summary(
    all_camera_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot BIC comparison across all cameras and models."""
    cameras = list(all_camera_results.keys())
    models = list(DISTORTION_MODELS.keys())

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(cameras))
    width = 0.15

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    for i, model in enumerate(models):
        bics = []
        for cam in cameras:
            model_result = all_camera_results[cam]['models'].get(model, {})
            bics.append(model_result.get('bic', np.nan))

        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, bics, width, label=model, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=45, ha='right')
    ax.set_ylabel('BIC Score')
    ax.set_title('Model Comparison Across All Cameras (BIC - lower is better)')
    ax.legend(title='Model')

    plt.tight_layout()

    for ext in ['png', 'svg']:
        fig.savefig(output_dir / f'bic_summary.{ext}')

    plt.close(fig)


def generate_recommendation(all_camera_results: Dict[str, Dict]) -> str:
    """Generate recommendation based on analysis results."""
    # Collect best models per camera
    best_models = []
    for cam, results in all_camera_results.items():
        bics = {m: results['models'][m]['bic'] for m in results['models'] if 'bic' in results['models'][m]}
        if bics:
            best = min(bics, key=bics.get)
            best_models.append(best)

    # Count votes
    from collections import Counter
    model_votes = Counter(best_models)

    # Check if zero model is adequate
    zero_adequate = True
    for cam, results in all_camera_results.items():
        zero_rms = results['models'].get('zero', {}).get('rms', 999)
        best_rms = min(r.get('rms', 999) for r in results['models'].values() if 'rms' in r)
        if zero_rms > best_rms * 1.5:  # Zero model 50% worse than best
            zero_adequate = False
            break

    # Check spatial pattern
    edge_vs_center_ratios = []
    for cam, results in all_camera_results.items():
        zero_spatial = results.get('spatial_errors', {}).get('zero', {})
        if zero_spatial:
            center = zero_spatial.get('center', {}).get('mean', 1)
            edges = []
            for region in ['top_center', 'bot_center', 'mid_left', 'mid_right']:
                if region in zero_spatial and zero_spatial[region].get('mean'):
                    edges.append(zero_spatial[region]['mean'])
            if edges and center > 0:
                edge_vs_center_ratios.append(np.mean(edges) / center)

    has_spatial_pattern = np.mean(edge_vs_center_ratios) > 1.5 if edge_vs_center_ratios else False

    # Generate recommendation
    if zero_adequate and not has_spatial_pattern:
        return (
            "RECOMMENDATION: Zero distortion model is adequate.\n"
            "Your cameras show minimal distortion. Continue using --fix_radial True --fix_tangential True\n"
            "This will provide the fastest and most stable calibration."
        )

    most_common_model = model_votes.most_common(1)[0][0] if model_votes else 'k1_k2'

    if most_common_model == 'k1_only':
        return (
            "RECOMMENDATION: Enable k1 radial distortion only.\n"
            "Your cameras have mild barrel/pincushion distortion.\n"
            "Use: --fix_radial False (and modify camera.py to only free k1)"
        )
    elif most_common_model in ['k1_k2', 'standard']:
        return (
            "RECOMMENDATION: Enable k1 and k2 radial distortion.\n"
            "Your cameras have moderate distortion that benefits from k1+k2 modeling.\n"
            "Use: --fix_radial False --fix_tangential True\n"
            "Consider modifying camera.py to fix k3 while freeing k1, k2."
        )
    elif most_common_model == 'full':
        return (
            "RECOMMENDATION: Enable full distortion model (k1, k2, k3, p1, p2).\n"
            "Your cameras show significant distortion including potential tangential components.\n"
            "Use: --fix_radial False --fix_tangential False"
        )
    else:
        return (
            "RECOMMENDATION: Enable k1 and k2 radial distortion (conservative default).\n"
            "Analysis was inconclusive. Starting with k1+k2 is a safe choice.\n"
            "Use: --fix_radial False --fix_tangential True"
        )


def diagnose_camera(
    camera_name: str,
    detections: List,
    boards: List,
    image_size: Tuple[int, int],
    n_samples: int,
    output_dir: Path
) -> Dict:
    """Run full diagnostic for a single camera."""
    print(f"\n{'='*60}")
    print(f"Diagnosing {camera_name}")
    print(f"{'='*60}")

    # Sample images for coverage
    selected_indices = sample_images_by_coverage(
        detections, boards, image_size, n_samples
    )
    print(f"Selected {len(selected_indices)} images for analysis")

    # Prepare calibration data
    selected_detections = [detections[i] for i in selected_indices]
    points = calibration_points(boards, selected_detections)

    if len(points.corners) < 10:
        print(f"WARNING: Only {len(points.corners)} valid detection frames. Need at least 10.")
        return {'error': 'Insufficient detections'}

    object_points = [np.array(pts, dtype=np.float32) for pts in points.object_points]
    image_points = [np.array(pts, dtype=np.float32) for pts in points.corners]

    print(f"Using {len(object_points)} images with {sum(len(p) for p in object_points)} total points")

    # Run calibration with each model
    results = {'models': {}, 'spatial_errors': {}}

    for model_name, config in DISTORTION_MODELS.items():
        print(f"  Testing model: {model_name}...", end=" ", flush=True)

        calib_result = calibrate_with_model(
            object_points, image_points, image_size, model_name
        )

        if calib_result['success']:
            # Compute BIC
            bic = compute_bic(
                calib_result['rms'],
                calib_result['n_points'],
                config['n_params']
            )
            calib_result['bic'] = bic

            # Compute spatial errors
            spatial = compute_spatial_errors(
                calib_result['K'],
                calib_result['dist'],
                calib_result['rvecs'],
                calib_result['tvecs'],
                object_points,
                image_points,
                image_size
            )
            results['spatial_errors'][model_name] = spatial

            print(f"RMS={calib_result['rms']:.3f}, BIC={bic:.1f}")

            # Plot spatial heatmap
            plot_spatial_heatmap(spatial, camera_name, model_name, output_dir)
        else:
            print(f"FAILED: {calib_result.get('error', 'unknown')}")

        results['models'][model_name] = calib_result

    # Plot model comparison for this camera
    successful_models = {m: r for m, r in results['models'].items() if r.get('success')}
    if successful_models:
        plot_model_comparison(successful_models, camera_name, output_dir)

    # Analyze distortion coefficients from full model
    full_result = results['models'].get('full', {})
    if full_result.get('success'):
        dist = full_result['dist']
        results['distortion_analysis'] = {
            'k1': {'value': float(dist[0]), 'significance': classify_coefficient(dist[0], 'k1')},
            'k2': {'value': float(dist[1]), 'significance': classify_coefficient(dist[1], 'k2')},
            'p1': {'value': float(dist[2]), 'significance': classify_coefficient(dist[2], 'p1')},
            'p2': {'value': float(dist[3]), 'significance': classify_coefficient(dist[3], 'p2')},
            'k3': {'value': float(dist[4]) if len(dist) > 4 else 0,
                   'significance': classify_coefficient(dist[4] if len(dist) > 4 else 0, 'k3')},
        }

    return results


def main():
    parser = ArgumentParser()
    parser.add_arguments(DiagnoseArgs, dest='args')
    program = parser.parse_args()
    args = program.args

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Distortion Diagnostic Tool")
    print(f"{'='*60}")
    print(f"Image path: {args.image_path}")
    print(f"Boards: {args.boards}")
    print(f"Sample images per camera: {args.sample_images}")
    print(f"Output directory: {output_dir}")

    # Load board configuration
    boards_dict = find_board_config(args.image_path, board_file=args.boards)
    board_names = list(boards_dict.keys())
    boards = list(boards_dict.values())
    print(f"Boards: {board_names}")

    # Find camera images
    camera_images = find_camera_images(
        args.image_path,
        args.cameras if args.cameras else None,
        args.camera_pattern,
        limit=None  # Load all, we'll sample later
    )

    camera_names = camera_images.cameras
    print(f"Found {len(camera_names)} cameras: {camera_names}")

    # Load images
    print(f"\nLoading images...")
    images = load_images(camera_images.filenames, j=args.num_threads, prefix=camera_images.image_path)

    # Get image sizes
    image_sizes = []
    for cam_images in images:
        if cam_images:
            h, w = cam_images[0].shape[:2]
            image_sizes.append((w, h))
        else:
            image_sizes.append((1920, 1080))  # Default

    # Detect boards
    print(f"\nDetecting boards...")
    detected_points = detect_images(boards, images, j=args.num_threads)

    # Run diagnostics per camera
    all_results = {}

    for cam_idx, camera_name in enumerate(camera_names):
        cam_detections = detected_points[cam_idx]
        cam_image_size = image_sizes[cam_idx]

        results = diagnose_camera(
            camera_name,
            cam_detections,
            boards,
            cam_image_size,
            args.sample_images,
            output_dir
        )

        all_results[camera_name] = results

    # Generate summary plots
    print(f"\n{'='*60}")
    print("Generating summary plots...")
    plot_coefficient_summary(all_results, output_dir)
    plot_bic_summary(all_results, output_dir)

    # Generate recommendation
    recommendation = generate_recommendation(all_results)
    print(f"\n{'='*60}")
    print(recommendation)
    print(f"{'='*60}")

    # Save JSON report
    report = {
        'cameras': {},
        'recommendation': recommendation,
        'summary': {
            'n_cameras': len(camera_names),
            'sample_images_per_camera': args.sample_images,
            'boards': board_names,
        }
    }

    for cam, results in all_results.items():
        cam_report = {
            'models': {},
            'distortion_analysis': results.get('distortion_analysis', {})
        }

        for model, model_result in results.get('models', {}).items():
            if model_result.get('success'):
                cam_report['models'][model] = {
                    'rms': model_result['rms'],
                    'bic': model_result.get('bic'),
                    'dist': model_result['dist'].tolist() if isinstance(model_result['dist'], np.ndarray) else model_result['dist'],
                    'n_images': model_result['n_images'],
                    'n_points': model_result['n_points']
                }

        # Add spatial errors for zero model
        if 'zero' in results.get('spatial_errors', {}):
            cam_report['spatial_errors_zero'] = {
                k: {'mean': v['mean'], 'n': v['n']}
                for k, v in results['spatial_errors']['zero'].items()
            }

        report['cameras'][cam] = cam_report

    report_path = output_dir / 'distortion_diagnosis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Plots saved to: {output_dir}")

    # Show interactive plots
    print("\nDisplaying interactive summary...")
    plt.ion()

    # Reopen coefficient summary for interactive viewing
    fig = plt.figure(figsize=(15, 10))
    img = plt.imread(output_dir / 'coefficient_summary.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Distortion Coefficients Summary (close to continue)')
    plt.show(block=True)


if __name__ == '__main__':
    main()
