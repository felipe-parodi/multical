# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multical is a multi-camera calibration library using one or more calibration patterns (charuco, aprilgrid, checkerboard). It supports both overlapping and non-overlapping camera scenarios via hand-eye calibration.

## Common Commands

```bash
# Install in development mode
uv pip install -e .

# Install with visualization support (qtpy, pyvistaqt)
uv pip install -e ".[interactive]"

# Run calibration
multical calibrate --image_path /path/to/images --boards board_config.yaml

# Run intrinsic-only calibration
multical intrinsic --input_path /path/to/images

# Visualize calibration results
multical vis --workspace_file calibration.pkl

# Test board detection on a single image
multical boards --boards my_board.yaml --detect my_image.jpeg

# Generate printable board image
multical boards --boards example_boards/charuco_16x22.yaml --paper_size A2 --pixels_mm 10 --write output_dir
```

## Architecture

### Entry Point and CLI (`multical/app/`)
- `multical.py` - Main CLI dispatcher with subcommands: `calibrate`, `intrinsic`, `boards`, `vis`
- `calibrate.py` - Multi-camera calibration workflow
- `intrinsic.py` - Single-camera intrinsic calibration
- Commands use dataclass-based argument parsing via `simple_parsing`

### Core Classes

**Workspace** (`multical/workspace.py`):
- Central orchestrator that manages the entire calibration pipeline
- Handles image loading, board detection, caching, calibration, and export
- Key methods: `add_camera_images()`, `detect_boards()`, `calibrate_single()`, `initialise_poses()`, `calibrate()`, `export()`

**Calibration** (`multical/optimization/calibration.py`):
- Holds optimization state: cameras, boards, poses, point tables
- Performs bundle adjustment via `scipy.optimize.least_squares`
- Key methods: `bundle_adjust()`, `reject_outliers()`, `adjust_outliers()`

**Camera** (`multical/camera.py`):
- Camera intrinsic model with OpenCV lens distortion
- Supported models: `standard`, `rational`, `thin_prism`, `tilted`, `full`
- Also supports fisheye via `camera_fisheye.py`

**Board** (`multical/board/`):
- `board.py` - Abstract base class
- `charuco.py` - ChArUco board implementation
- `aprilgrid.py` - AprilGrid board (Kalibr-compatible)

### Configuration (`multical/config/`)
- `arguments.py` - Dataclass definitions for CLI options (`PathOpts`, `CameraOpts`, `RuntimeOpts`, `OptimizerOpts`)
- `workspace.py` - Helper functions: `initialise_with_images()`, `optimize()`

### Transforms (`multical/transform/`)
- `rtvec.py` - Rotation vector utilities
- `matrix.py` - 4x4 homogeneous transformation matrices
- `hand_eye.py` - Hand-eye calibration transforms

### Motion Models (`multical/motion/`)
- `static_frames.py` - Static camera rig assumption
- `rolling_frames.py` - Rolling shutter support

## Board Configuration Format

YAML files in `example_boards/`:
```yaml
boards:
  board_name:
    _type_: charuco  # or aprilgrid, checkerboard
    size: [16, 22]   # columns, rows
    aruco_dict: 4X4_1000
    square_length: 0.025  # meters
    marker_length: 0.01875
    min_rows: 3
    min_points: 20
```

## Output Files

- `calibration.json` - Camera intrinsics, relative poses, rig poses
- `calibration.pkl` - Full workspace state for visualization/resumption
- `calibration.detections.pkl` - Cached board detections
- `calibration.txt` - Log file

## Key Dependencies

- OpenCV (opencv-contrib-python) - Detection, calibration primitives
- scipy - Bundle adjustment optimizer
- py-structs - Data structure utilities
- numpy-quaternion - Rotation representations
