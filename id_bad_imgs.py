import pickle
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import collections

# Attempt to import from the local multical package
try:
    from multical import tables
    # Try importing Calibration from workspace first
    try:
        from multical.workspace import Calibration
    except ImportError:
        # If not in workspace, maybe it's directly under multical?
        from multical import Calibration # Requires it to be in __init__.py

except ImportError as e:
    print(f"Error importing multical components: {e}")
    print("Please ensure the script is run from a location where the local 'multical' package is discoverable,")
    print("or that the 'multical' package is correctly installed in your environment.")
    exit()

# --- Configuration ---
# Path to the multical results file
pkl_calib_results = r"A:\EnclosureProjects\inprep\freemat\experiments\good\240528\video\calibration\multical\images\half_cams\half_std_soft_l1_fixed_dist_board0.pkl"

# --- Visualization Configuration ---
# Path to the directory containing camera image folders (e.g., Cam_001, Cam_002)
IMAGE_BASE_DIR = r"A:\EnclosureProjects\inprep\freemat\experiments\good\240528\video\calibration\multical\images\half_cams"
# --- Visualization Output Dir Calculation --- 
output_dirname = os.path.splitext(os.path.basename(pkl_calib_results))[0] + "_outlier_viz"
VIZ_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(pkl_calib_results)), output_dirname)
# --- Analysis Configuration ---
# Number of top individual outliers to display
N_TOP_OUTLIERS = 20
# Number of top frames (by max error) to analyze/visualize
N_TOP_FRAMES = 5
# Number of top cameras (by error within a frame) to visualize per frame
N_TOP_CAMERAS_PER_FRAME = 3

# --- Functions ---

def load_multical_results(filepath: str):
    """Loads the calibration Workspace object from a .pkl file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully loaded data from {filepath}")
            print(f"Loaded data type: {type(data)}")

            # Check if it's a Workspace object
            if 'Workspace' in str(type(data)):
                 print("Loaded Workspace object directly.")
                 # Verify essential attributes exist for our purpose
                 required_attrs = ['calibrations', 'point_table', 'filenames', 'names'] # Changed 'cameras' to 'names'
                 if all(hasattr(data, attr) for attr in required_attrs):
                     # Perform a basic check on the first calibration object if possible
                     if isinstance(data.calibrations, dict) and data.calibrations:
                          first_calib = next(iter(data.calibrations.values()), None)
                          if isinstance(first_calib, Calibration):
                              print("Workspace contains valid Calibration object(s).")
                              return data # Return the whole workspace
                          else:
                              print("Error: First item in workspace.calibrations is not a Calibration object.")
                              return None
                     elif isinstance(data.calibrations, list) and data.calibrations: # Handle list case too
                          if isinstance(data.calibrations[0], Calibration):
                              print("Workspace contains valid Calibration object(s).")
                              return data # Return the whole workspace
                          else:
                              print("Error: First item in workspace.calibrations list is not a Calibration object.")
                              return None
                     else:
                         print("Error: workspace.calibrations is not a non-empty dict or list.")
                         return None
                 else:
                     print(f"Error: Workspace object is missing one or more required attributes: {required_attrs}")
                     return None
            else:
                 print("Error: Loaded data is not a Workspace object.")
                 return None

    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle data from {filepath}. File might be corrupted or incompatible.")
        return None
    except ImportError:
         # Catch potential issues if the pickled object's class isn't found
         print(f"Error: Could not import a class required by the pickled object in {filepath}.")
         print("Ensure the necessary version of 'multical' is available in the environment.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None

def get_outlier_info(calib: Calibration, camera_names_struct: object, num_outliers: int = 10):
    """Calculates reprojection errors and returns info about the top outliers."""
    # Convert camera_names struct to list for easier indexing
    try:
        # Access the list of camera names under the .camera attribute
        camera_names = list(camera_names_struct.camera)
    except (TypeError, AttributeError) as e:
        print(f"Warning: Could not get camera names list from camera_names_struct.camera: {e}")
        camera_names = [] # Fallback

    if not hasattr(calib, 'reprojected') or not hasattr(calib, 'point_table'):
        print("Error: Calibration object missing 'reprojected' or 'point_table' attributes.")
        return []

    try:
        # Calculate errors for ALL points first
        errors_all, valid_all = tables.reprojection_error(calib.reprojected, calib.point_table)
        print(f"Calculated all reprojection errors. Shape: {errors_all.shape}, Total valid points initially: {np.sum(valid_all)}")

        # Check if an inlier mask exists and apply it
        if hasattr(calib, 'inliers') and isinstance(calib.inliers, np.ndarray) and calib.inliers.shape == valid_all.shape:
            print(f"Applying final inlier mask. Total inliers in mask: {np.sum(calib.inliers)}")
            final_valid_mask = valid_all & calib.inliers
        else:
            print("Warning: Final inlier mask not found or incompatible. Using only reprojection validity.")
            final_valid_mask = valid_all

        print(f"Analyzing {np.sum(final_valid_mask)} points after applying final mask.")

        if not np.any(final_valid_mask):
             print("No valid points remain after applying the final mask.")
             return []

        # Get errors and indices ONLY for the final valid set
        valid_errors = errors_all[final_valid_mask]
        valid_indices_tuple = np.where(final_valid_mask)

        # Find the indices that would sort the valid errors in descending order
        sorted_indices_desc = np.argsort(valid_errors)[::-1]

        # Get the top N outlier indices relative to the *flattened* valid arrays
        top_n_indices_flat = sorted_indices_desc[:num_outliers]

        outlier_metadata = []
        print(f"\n--- Top {min(num_outliers, len(top_n_indices_flat))} Outliers ---")
        for i, flat_idx in enumerate(top_n_indices_flat):
            original_indices = None # Initialize outside try block
            try:
                # Map the flat index back to the original multidimensional index tuple
                original_indices = tuple(dim_array[flat_idx] for dim_array in valid_indices_tuple)
                cam_idx, frame_idx, board_idx, point_idx = original_indices

                error_value = errors_all[original_indices]

                # Retrieve metadata using the converted list
                cam_name = camera_names[cam_idx] if camera_names and len(camera_names) > cam_idx else f"Cam_{cam_idx}"
                
                # More robust check for board name
                board_name = f"Board_{board_idx}" # Default value
                if hasattr(calib, 'boards') and calib.boards:
                    if hasattr(calib.boards, 'names') and calib.boards.names and len(calib.boards.names) > board_idx:
                        board_name = calib.boards.names[board_idx]
                    else:
                        # Pass silently for now, default name is used
                        # print(f"Warning: calib.boards found, but 'names' attribute missing, empty, or too short for board_idx {board_idx}.")
                        pass
                else:
                    # Pass silently for now, default name is used
                    # print("Warning: calib.boards attribute missing or empty.")
                    pass

                uv_detected = calib.point_table.points[original_indices]
                # corner_id = calib.point_table.ids[original_indices] # If corner IDs are stored

                metadata = {
                    'rank': i + 1,
                    'error': error_value,
                    'camera': cam_name,
                    'frame': frame_idx,
                    'board': board_name,
                    'point_idx_in_board': point_idx, # Index within the detected points for that board/frame/cam
                    # 'corner_id': corner_id, # Uncomment if available
                    'uv_detected': uv_detected.tolist(), # Convert numpy array to list for printing
                    'original_indices': original_indices
                }
                outlier_metadata.append(metadata)

                # Print concise info
                print(f"  {metadata['rank']}. Error: {metadata['error']:.4f} | "
                    f"Cam: {metadata['camera']} | Frame: {metadata['frame']} | "
                    f"Board: {metadata['board']} | PtIdx: {metadata['point_idx_in_board']} | "
                    f"UV: ({metadata['uv_detected'][0]:.1f}, {metadata['uv_detected'][1]:.1f})")

            except Exception as e:
                print(f"\n!!! Error processing outlier rank {i+1} (flat_idx: {flat_idx}) !!!")
                if original_indices:
                    print(f"  Indices causing error: {original_indices}")
                else:
                    print("  Error occurred before indices could be determined.")
                print(f"  Error Type: {type(e)}")
                print(f"  Error Details: {e}")
                import traceback
                traceback.print_exc() # Print full traceback
                print("  Skipping remaining outliers.")
                break # Stop processing further outliers

        return outlier_metadata

    except AttributeError as e:
        print(f"Error accessing attributes during error calculation: {e}")
        print("The structure of the loaded Calibration object might be different than expected.")
        return []
    except IndexError as e:
        print(f"Error accessing data using indices: {e}")
        print("Mismatch between indices and data dimensions?")
        return []
    except Exception as e: # Broad exception catch, refine if needed
        print(f"Error during outlier analysis: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Visualization Function ---
def visualize_frame_detections(workspace: object, frame_idx: int, cam_names: list[str],
                             image_base_dir: str, output_dir: str):
    """Loads images for a specific frame and cameras, draws detected and reprojected points."""
    print(f"\n--- Starting Visualization for Frame {frame_idx} ---")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualization images to: {output_dir}")

    # We need the Calibration object for reprojections
    # Assuming it's the first one in the ordered dict `workspace.calibrations`
    if not (workspace.calibrations and isinstance(workspace.calibrations, collections.OrderedDict)):
         print("Error: workspace.calibrations is not a valid OrderedDict.")
         return
    calib_obj = next(iter(workspace.calibrations.values()))
    if not isinstance(calib_obj, Calibration):
         print("Error: Could not get valid Calibration object from workspace.")
         return

    # Get camera name to index mapping
    try:
        # Use the .camera attribute from the struct
        cam_name_to_idx = {name: i for i, name in enumerate(workspace.names.camera)}
    except AttributeError:
        print("Error: Could not get camera names from workspace.names.camera") # Updated error message
        return

    for cam_name in cam_names:
        if cam_name not in cam_name_to_idx:
            print(f"Warning: Camera '{cam_name}' not found in workspace camera list. Skipping visualization.")
            continue

        cam_idx = cam_name_to_idx[cam_name]

        # --- Find the image filename --- 
        img_filename = None
        try:
            # Check if filenames is a list of lists/tuples (cam_idx, frame_idx)
            if isinstance(workspace.filenames, (list, tuple)) and len(workspace.filenames) > cam_idx and \
               isinstance(workspace.filenames[cam_idx], (list, tuple)) and len(workspace.filenames[cam_idx]) > frame_idx:
                img_filename = workspace.filenames[cam_idx][frame_idx]
                if not isinstance(img_filename, str):
                     print(f"Warning: Found filename for {cam_name}, frame {frame_idx} is not a string ({type(img_filename)}). Skipping.")
                     img_filename = None # Reset
            # Add other potential structures for workspace.filenames if needed
            # elif isinstance(workspace.filenames, dict) ...
            else:
                 print(f"Warning: workspace.filenames structure not recognized or index out of bounds for {cam_name}, frame {frame_idx}. Attempting fallback naming.")
                 # Fallback: Assume standard naming like frame_0XXX.png/jpg in cam folder
                 cam_image_dir = os.path.join(image_base_dir, cam_name)
                 potential_png = os.path.join(cam_image_dir, f"frame_{frame_idx:04d}.png")
                 potential_jpg = os.path.join(cam_image_dir, f"frame_{frame_idx:04d}.jpg")
                 if os.path.exists(potential_png):
                     img_filename = potential_png
                 elif os.path.exists(potential_jpg):
                      img_filename = potential_jpg
                 else:
                      print(f"Error: Could not find image file for {cam_name}, frame {frame_idx} using fallback naming in {cam_image_dir}")
                      continue # Skip this camera
        except Exception as e:
             print(f"Error accessing workspace.filenames for {cam_name}, frame {frame_idx}: {e}. Attempting fallback.")
             # Fallback logic repeated here for safety, can be refactored
             cam_image_dir = os.path.join(image_base_dir, cam_name)
             potential_png = os.path.join(cam_image_dir, f"frame_{frame_idx:04d}.png")
             potential_jpg = os.path.join(cam_image_dir, f"frame_{frame_idx:04d}.jpg")
             if os.path.exists(potential_png):
                 img_filename = potential_png
             elif os.path.exists(potential_jpg):
                  img_filename = potential_jpg
             else:
                  print(f"Error: Could not find image file for {cam_name}, frame {frame_idx} using fallback naming after error.")
                  continue

        if not img_filename or not os.path.exists(img_filename):
            print(f"Error: Image file path '{img_filename}' not found or invalid for {cam_name}, frame {frame_idx}. Skipping.")
            continue

        # --- Load Image --- 
        print(f"  Loading image: {img_filename}")
        img = cv2.imread(img_filename)
        if img is None:
            print(f"Error: Failed to load image {img_filename}. Skipping {cam_name}.")
            continue
        img_vis = img.copy() # Work on a copy

        # --- Get Points and Validity --- 
        try:
            # <<< Inspect Data Structures >>>
            print(f"    Inspecting workspace.point_table: type={type(workspace.point_table)}")
            if hasattr(workspace.point_table, '__dict__'): print(f"      point_table attributes: {list(workspace.point_table.__dict__.keys())}")
            if hasattr(workspace.point_table, 'points'): print(f"      point_table.points: type={type(workspace.point_table.points)}")
            if hasattr(workspace.point_table, 'valid'): print(f"      point_table.valid: type={type(workspace.point_table.valid)}")
            
            print(f"    Inspecting calib_obj.reprojected: type={type(calib_obj.reprojected)}")
            if hasattr(calib_obj.reprojected, '__dict__'): print(f"      reprojected attributes: {list(calib_obj.reprojected.__dict__.keys())}")
            # <<< End Inspection >>>

            # Assuming only one board (board_idx=0)
            board_idx = 0 
            # !! Access might need adjustment based on inspection !!
            valid_mask = workspace.point_table.valid[cam_idx, frame_idx, board_idx, :]
            uv_detected = workspace.point_table.points[cam_idx, frame_idx, board_idx, :, :]
            # Try accessing reprojected points via a .points attribute, similar to point_table
            if hasattr(calib_obj.reprojected, 'points') and isinstance(calib_obj.reprojected.points, np.ndarray):
                 uv_reprojected = calib_obj.reprojected.points[cam_idx, frame_idx, board_idx, :, :]
            else:
                 # Fallback/Error - If .points doesn't exist or isn't an array
                 print(f"Error: Cannot find suitable numpy array in calib_obj.reprojected for {cam_name}, frame {frame_idx}. Skipping point drawing for this camera.")
                 # We could try other attributes like .values if needed
                 continue # Skip drawing for this camera

        except (IndexError, AttributeError) as e:
            print(f"Error accessing point data for {cam_name}, frame {frame_idx}: {e}. Skipping.")
            continue

        # --- Draw Points --- 
        n_drawn = 0
        for pt_idx in range(len(valid_mask)):
            if valid_mask[pt_idx]:
                # Detected point (blue circle)
                center_det = tuple(np.round(uv_detected[pt_idx]).astype(int))
                cv2.circle(img_vis, center_det, 5, (255, 0, 0), 1) # Blue

                # Reprojected point (red cross)
                center_rep = tuple(np.round(uv_reprojected[pt_idx]).astype(int))
                cv2.drawMarker(img_vis, center_rep, (0, 0, 255), cv2.MARKER_CROSS, 10, 1) # Red

                # Draw line between them (yellow)
                cv2.line(img_vis, center_det, center_rep, (0, 255, 255), 1)
                
                # --- Temporarily disable error calculation/drawing ---
                # # Optional: Add error value as text
                # error_val = point_errors[pt_idx]
                # if error_val > 2.0: # Only label large errors
                #      cv2.putText(img_vis, f"{error_val:.1f}", (center_det[0] + 5, center_det[1] - 5),
                #                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1) # Green text
                n_drawn += 1

        print(f"    Drew {n_drawn} valid points for {cam_name}.")

        # --- Save Image --- 
        output_filename = f"outliers_{cam_name}_frame_{frame_idx}.png"
        output_filepath = os.path.join(output_dir, output_filename)
        try:
            cv2.imwrite(output_filepath, img_vis)
            print(f"    Saved visualization to {output_filepath}")
        except Exception as e:
            print(f"Error saving image {output_filepath}: {e}")

    print("--- Visualization Finished ---")


# --- Main execution ---
if __name__ == "__main__":
    print(f"Analyzing calibration results from: {pkl_calib_results}")

    # Load the full workspace
    workspace = load_multical_results(pkl_calib_results)

    if workspace:
        # <<< Inspect workspace.names >>>
        print(f"Inspecting workspace.names: type={type(workspace.names)}, value={workspace.names}")
        # <<< End Inspection >>>

        # --- Extract the primary Calibration object --- 
        # Assuming the first one in the OrderedDict is the one we optimized/care about
        if isinstance(workspace.calibrations, dict) and workspace.calibrations:
            calib_obj = next(iter(workspace.calibrations.values()))
            print(f"Extracted Calibration object (type: {type(calib_obj)}). Proceeding with analysis.")
        elif isinstance(workspace.calibrations, list) and workspace.calibrations: # Handle list case
             calib_obj = workspace.calibrations[0]
             print(f"Extracted Calibration object from list (type: {type(calib_obj)}). Proceeding with analysis.")
        else:
            print("Error: Could not extract a Calibration object from workspace.calibrations.")
            calib_obj = None

        if calib_obj and isinstance(calib_obj, Calibration):
            # Pass workspace.names to the function
            outliers = get_outlier_info(calib_obj, workspace.names, num_outliers=N_TOP_OUTLIERS)

            if outliers: # Only proceed if we have outliers
                # --- Outlier Counts --- 
                # (Counts per frame/camera for top N individual outliers)
                frame_counts = {}
                for outlier in outliers:
                    frame = outlier['frame']
                    frame_counts[frame] = frame_counts.get(frame, 0) + 1
                print("\n--- Outlier Counts per Frame (Top N Outliers) ---")
                sorted_frames_by_count = sorted(frame_counts.items(), key=lambda item: item[1], reverse=True)
                for frame, count in sorted_frames_by_count:
                    print(f"  Frame {frame}: {count} outlier(s)")

                cam_counts = {}
                for outlier in outliers:
                    cam = outlier['camera']
                    cam_counts[cam] = cam_counts.get(cam, 0) + 1
                print("\n--- Outlier Counts per Camera (Top N Outliers) ---")
                sorted_cams_by_count = sorted(cam_counts.items(), key=lambda item: item[1], reverse=True)
                for cam, count in sorted_cams_by_count:
                    print(f"  Camera {cam}: {count} outlier(s)")
                # --- End Outlier Counts ---

                # --- Analyze Top Frames by Max Error ---
                print(f"\n--- Analyzing Top {N_TOP_FRAMES} Frames by Maximum Error ---")
                top_frames_to_visualize = [] # Store the indices of frames to visualize
                try:
                    all_errors, all_valid = tables.reprojection_error(calib_obj.reprojected, workspace.point_table)
                    frame_max_errors = {}
                    valid_indices_tuple = np.where(all_valid)

                    if valid_indices_tuple[0].size > 0:
                        frame_indices = valid_indices_tuple[1]
                        valid_error_values = all_errors[all_valid]
                        unique_frames = np.unique(frame_indices)
                        for frame_idx_iter in unique_frames:
                            errors_for_frame = valid_error_values[frame_indices == frame_idx_iter]
                            if errors_for_frame.size > 0:
                                frame_max_errors[frame_idx_iter] = np.max(errors_for_frame)
                            else:
                                frame_max_errors[frame_idx_iter] = 0
                        
                        sorted_frame_errors = sorted(frame_max_errors.items(), key=lambda item: item[1], reverse=True)

                        print(f"--- Top {min(N_TOP_FRAMES, len(sorted_frame_errors))} Frames by Max Reprojection Error ---")
                        for i in range(min(N_TOP_FRAMES, len(sorted_frame_errors))):
                            frame_idx_iter, max_err = sorted_frame_errors[i]
                            print(f"  {i+1}. Frame {frame_idx_iter}: Max Error = {max_err:.4f}")
                            top_frames_to_visualize.append(frame_idx_iter) # Add frame to visualization list
                    else:
                        print("No valid points found to analyze per-frame errors.")
                except Exception as e:
                    print(f"Error during per-frame error analysis: {e}")
                # --- End Per-Frame Analysis ---

                # --- Dynamic Visualization --- 
                if top_frames_to_visualize:
                    if IMAGE_BASE_DIR and os.path.isdir(IMAGE_BASE_DIR):
                        print(f"\n--- Preparing Visualization for Top {len(top_frames_to_visualize)} Frames ---")
                        # Ensure we have the full error and index data
                        if 'all_errors' not in locals() or 'all_valid' not in locals():
                             print("Error: Full error data not available for dynamic camera selection.")
                        else:
                             # --- Corrected Assignments ---
                             # valid_indices_tuple was calculated earlier from np.where(all_valid)
                             # all_errors[all_valid] was calculated earlier and stored in valid_error_values in the previous block
                             if 'valid_indices_tuple' not in locals() or 'valid_error_values' not in locals():
                                 print("Error: Required index/error variables not in scope for dynamic visualization.")
                             else:
                                 valid_cam_indices = valid_indices_tuple[0]  # Direct assignment (already filtered by where)
                                 valid_frame_indices = valid_indices_tuple[1] # Direct assignment (already filtered by where)
                                 # valid_error_values is already defined from the previous block

                                 for frame_to_viz in top_frames_to_visualize:
                                     print(f"\nProcessing visualization for Frame {frame_to_viz}...")
                                     # Find cameras with top errors *in this frame*
                                     frame_mask = (valid_frame_indices == frame_to_viz)
                                     # Use the pre-filtered valid_error_values
                                     errors_in_frame = valid_error_values[frame_mask]
                                     # Use the pre-filtered valid_cam_indices
                                     cams_in_frame = valid_cam_indices[frame_mask]
                                     
                                     cameras_to_visualize_for_frame = []
                                     if errors_in_frame.size > 0:
                                         # Get indices that sort errors descending
                                         sorted_error_indices_in_frame = np.argsort(errors_in_frame)[::-1]
                                         # Get camera indices corresponding to top N errors
                                         top_cam_indices = cams_in_frame[sorted_error_indices_in_frame[:N_TOP_CAMERAS_PER_FRAME]]
                                         unique_top_cam_indices = np.unique(top_cam_indices)
                                         # Convert indices to names
                                         try:
                                             cameras_to_visualize_for_frame = [workspace.names.camera[idx] for idx in unique_top_cam_indices]
                                             print(f"  Selected cameras for Frame {frame_to_viz} (Top {N_TOP_CAMERAS_PER_FRAME} errors): {cameras_to_visualize_for_frame}")
                                         except (AttributeError, IndexError, TypeError) as e:
                                              print(f"  Error converting top camera indices to names for Frame {frame_to_viz}: {e}")
                                              cameras_to_visualize_for_frame = [] # Skip if conversion fails
                                     else:
                                          print(f"  No valid errors found in Frame {frame_to_viz} to select cameras.")
                                      
                                     # Call visualization if we have cameras selected
                                     if cameras_to_visualize_for_frame:
                                         visualize_frame_detections(workspace, frame_to_viz,
                                                                cameras_to_visualize_for_frame, IMAGE_BASE_DIR,
                                                                VIZ_OUTPUT_DIR)
                                     else:
                                         print(f"  Skipping visualization for Frame {frame_to_viz} due to lack of selected cameras.")
                    else:
                        print("\nSkipping visualization: IMAGE_BASE_DIR is not set or not a valid directory.")
                else:
                    print("\nSkipping visualization: No top frames identified for visualization.")
            else:
                print("\nNo outliers identified, skipping further analysis and visualization.")
        else:
            print("Could not obtain a valid Calibration object from the workspace.")

    else:
        print("Could not load or validate the Workspace object. Exiting.") 