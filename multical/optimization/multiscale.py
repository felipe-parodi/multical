# from typing import List, Tuple
# import numpy as np
# import cv2
# from ..camera import Camera, calibration_points, top_detection_coverage
# from structs.struct import struct
# from ..io.logging import info

# def scale_points(points, scale: float):
#     """Scale detection points by given factor"""
#     return points._map(lambda corners: corners * scale if corners is not None else None)

# def scale_image_size(image_size: Tuple[int, int], scale: float) -> Tuple[int, int]:
#     """Scale image size maintaining aspect ratio"""
#     return tuple(int(s * scale) for s in image_size)

# def scale_camera_params(camera: Camera, scale: float) -> Camera:
#     """Scale camera intrinsics for different image size"""
#     K = camera.intrinsic.copy()
#     K[0:2, :] *= scale
#     return camera.copy(intrinsic=K)

# def multiscale_calibrate(boards, detections, image_size, intrinsic_error_limit,
#                         max_iter=10, eps=1e-3, model='standard',
#                         fix_aspect=False, has_skew=False, flags=0, max_images=None,
#                         scales=[0.25, 0.5, 1.0]):
#     """
#     Multi-scale camera calibration implementation
#     """
#     camera = None
#     final_err = intrinsic_error_limit

#     for scale in scales:
#         info(f"Calibrating at scale {scale:.2f}")
#         scaled_size = scale_image_size(image_size, scale)
#         scaled_points = scale_points(detections, scale)

#         # Scale error limit proportionally
#         scaled_error_limit = intrinsic_error_limit * scale

#         # Use previous estimate as starting point if available
#         if camera is not None:
#             scaled_camera = scale_camera_params(camera, scale)
#             kwargs = {
#                 'K': scaled_camera.intrinsic,
#                 'D': camera.dist
#             }
#         else:
#             kwargs = {}

#         # Calibrate at current scale
#         camera, err = Camera.calibrate(
#             boards=boards,
#             detections=scaled_points,
#             image_size=scaled_size,
#             intrinsic_error_limit=scaled_error_limit,
#             max_iter=max_iter,
#             eps=eps,
#             model=model,
#             fix_aspect=fix_aspect,
#             has_skew=has_skew,
#             flags=flags,
#             max_images=max_images,
#             **kwargs
#         )
#         final_err = err / scale  # Scale error back

#     # Scale camera parameters back to original size
#     if camera is not None:
#         camera = scale_camera_params(camera, 1.0/scales[-1])

#     return camera, final_err
