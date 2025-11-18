import pickle
from multical.io.logging import info
import os
from structs.struct import struct


def try_load_detections(filename, cache_key={}):
    try:
        with open(filename, "rb") as file:
            loaded = pickle.load(file)

            # Check that the detections match the metadata
            if loaded.get("cache_key", {}) == cache_key or check_dataset_similarity(
                loaded, cache_key
            ):
                info(f"Loaded detections from {filename}")
                return loaded.detected_points
            else:
                info(f"Config changed, not using loaded detections in {filename}")
    except (OSError, IOError, EOFError, AttributeError) as e:
        return None


def try_load_cache_data(filename):
    """Load the full cache data structure (not just detected_points)"""
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except (OSError, IOError, EOFError, AttributeError) as e:
        return None


def check_dataset_similarity(loaded, cache_key):
    """
    Checks whether the loaded datasets file format is similar to cached dataset.
    Returns False if datasets don't match (different number of cameras, images, or paths).
    This allows partial cache reuse to be attempted.
    """
    filenames = loaded.cache_key["filenames"]
    caches = cache_key["filenames"]

    # Check if number of cameras matches
    if len(filenames) != len(caches):
        info(f"Cache has {len(filenames)} cameras but current dataset has {len(caches)} cameras - will attempt partial cache reuse")
        return False

    # Check each camera's image count
    for i in range(len(filenames)):
        if len(filenames[i]) != len(caches[i]):
            info(f"Camera {i}: cache has {len(filenames[i])} images but current dataset has {len(caches[i])} images - will attempt partial cache reuse")
            return False

        # Check if image paths are similar (last 3 directory components)
        for j in range(len(filenames[i])):
            file_dirs = find_char(filenames[i][j])
            cache_dirs = find_char(caches[i][j])
            if file_dirs[-3:] != cache_dirs[-3:]:
                return False

    return True


def find_char(str):
    """
    Helper function for check_dataset_similarity function
    """
    x = str.split("\\")
    y = str.split("/")
    return x if len(x) > len(y) else y


def write_detections(filename, detected_points, cache_key={}):
    data = struct(cache_key=cache_key, detected_points=detected_points)
    with open(filename, "wb") as file:
        pickle.dump(data, file)
