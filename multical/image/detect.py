from functools import partial
import os.path as path
import cv2

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import  ThreadPool                                         
from tqdm import tqdm
import numpy as np
from multical.camera import  stereo_calibrate
from multical.threading import parmap_lists
from structs.struct import struct as Struct

from structs.struct import transpose_structs, filter_none

# def load_image(filename):
#   assert path.isfile(filename), f"load_image: file {filename} does not exist"

#   image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#   assert image is not None, f"load_image: could not read {filename}"
#   return image
def load_image(filename):
    assert path.isfile(filename), f"load_image: file {filename} does not exist"
    
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"load_image: could not read {filename}"
    
    # Add explicit conversion to numpy array and type checking
    image = np.asarray(image)
    return image

  

def common_image_size(images):
  image_shape = images[0].shape
  h, w, *_ = image_shape

  assert all([image.shape == image_shape for image in images])
  return (w, h)


def load_images(filenames, prefix=None, **map_options):
    if prefix is not None:
      filenames = [[path.join(prefix, file) for file in camera_files]
        for camera_files in filenames]

    return parmap_lists(load_image, filenames, **map_options)

# def detect_image(image, boards):
#     try:
#         return [board.detect(image) for board in boards]
#     except Exception as e:
#         print(f"Error in detect_image: {e}")
#         return None
def detect_image(image, boards):
    try:
        if image is None:
            print("Error: Image is None")
            return None
            
        if not isinstance(image, np.ndarray):
            print(f"Error: Image is not numpy array, type is {type(image)}")
            return None
            
        if len(image.shape) != 2:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                print(f"Error: Invalid image shape {image.shape}")
                return None
                
        return [board.detect(image) for board in boards]
        
    except Exception as e:
        print(f"Error in detect_image: {e}")
        return None

def detect_images(boards, images, j=None, **map_options):
    detect = partial(detect_image, boards=boards)
    
    # Calculate optimal thread distribution
    total_images = sum(len(image_set) for image_set in images)
    num_cameras = len(images)
    
    # Use about 75% of threads for image processing, rest for camera parallelization
    if j is None:
        j = cpu_count()
    threads_per_camera = max(1, (j * 3) // (num_cameras * 4))
    camera_threads = max(1, j // 4)
    print("DO YOU SEE THIS?") # yes
    print(f"Using {camera_threads} camera threads with {threads_per_camera} threads per camera")
    
    def process_camera(camera_images):
        with ThreadPool(processes=threads_per_camera) as pool:
            return list(pool.imap(detect, camera_images))
    
    with ThreadPool(processes=camera_threads) as pool:
        with tqdm(total=total_images, desc="Detecting boards", unit="img") as pbar:
            results = []
            for camera_results in pool.imap(process_camera, images):
                results.append(camera_results)
                pbar.update(len(camera_results))
                
    return results
# def detect_images(boards, images, **map_options):
#     print("Starting detect_images")
#     detect = partial(detect_image, boards=boards)
#     print(f"Processing {len(images)} images")
#     try:
#         results = parmap_lists(detect, images, **map_options, pool=Pool)
#         print("Detection complete")
#         return results
#     except Exception as e:
#         print(f"Error in detect_images: {e}")
#         raise


def intersect_detections(board, d1, d2):
  ids, inds1, inds2 = np.intersect1d(d1.ids, d2.ids, return_indices=True)

  if len(ids) > 0:
    return struct(points1 = d1.corners[inds1], points2 = d2.corners[inds2], 
      object_points = board.points[ids], ids=ids)
  else:
    return None

def stereo_calibrate_detections(detections, board, cameras, i, j, **kwargs):
  matching = [intersect_detections(board, d1, d2) 
    for d1, d2 in zip(detections[i], detections[j])]

  matching_frames = transpose_structs(filter_none(matching))
  return stereo_calibrate((cameras[i], cameras[j]), matching_frames, **kwargs)  
  


      
