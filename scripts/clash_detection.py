import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Any

# --- Constants ---
GEOMETRY_COLUMNS = ['GlobalId', 'x', 'y', 'z', 'width', 'depth', 'height']
GLOBAL_ID_COL = 'GlobalId'
X_COL = 'x'
Y_COL = 'y'
Z_COL = 'z'
WIDTH_COL = 'width'
DEPTH_COL = 'depth'
HEIGHT_COL = 'height'

DATA_DIR = "data"
OUTPUT_DIR = "output" # Changed from 'data' to avoid cluttering input data

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Get Project Root ---
# --- Helper Function: Get Project Root ---
def get_project_root() -> str:
    """Gets the project root directory based on the script's location."""
    # Assuming the script is in a 'scripts' or 'src' directory
    # Adjust the number of '..' based on your actual structure
    # If script is at root, use os.path.dirname(os.path.abspath(_file)) # Use __file_
    try:
        # This works when running as a script
        script_dir = os.path.dirname(__file__) # CORRECTED: Double underscores
        return os.path.abspath(os.path.join(script_dir, '..'))
    except NameError:
        # Fallback for interactive environments (like Jupyter)
        # Assumes the notebook/interactive session is running from the project root
        logging.warning("_file_ not defined. Assuming current working directory is project root.") # Corrected message too
        return os.path.abspath('.')

# --- Function to Load Data using Pandas ---
def load_csv_data(file_path: str, file_description: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file using pandas.

    Args:
        file_path: Path to the CSV file.
        file_description: A description of the file being loaded (for logging).

    Returns:
        A pandas DataFrame containing the loaded data, or None if loading fails.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"{file_description} file not found at: {file_path}")
            return None
        df = pd.read_csv(file_path)
        logging.info(f"{file_description} loaded successfully from '{os.path.basename(file_path)}'. Found {len(df)} rows.")
        # Removed df.fillna('N/A') - handle missing/invalid values during computation
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"Error loading {file_description} file {file_path}: File is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading {file_description} file {file_path}: {e}")
        return None

# --- Function to Compute Bounding Box for an Element ---
def compute_bounding_box(row: pd.Series) -> Optional[Dict[str, float]]:
    """
    Computes the Axis-Aligned Bounding Box (AABB) for an element row.

    Args:
        row: A pandas Series representing a single element, expected to contain
             geometry columns defined in GEOMETRY_COLUMNS.

    Returns:
        A dictionary representing the bounding box {'min_x', 'max_x', ...},
        or None if geometry data is invalid or missing.
    """
    if row is None:
        logging.warning("compute_bounding_box received None input row.")
        return None

    global_id = row.get(GLOBAL_ID_COL, 'Unknown')

    # Check if required columns exist in the row (though check in main is primary)
    for col in [X_COL, Y_COL, Z_COL, WIDTH_COL, DEPTH_COL, HEIGHT_COL]:
        if col not in row:
            logging.error(f"Missing column '{col}' for element {global_id}.")
            return None

    try:
        # Check for NaN or None before conversion
        if pd.isna(row[X_COL]) or pd.isna(row[Y_COL]) or pd.isna(row[Z_COL]) or \
           pd.isna(row[WIDTH_COL]) or pd.isna(row[DEPTH_COL]) or pd.isna(row[HEIGHT_COL]):
             logging.warning(f"Missing geometry value(s) for element {global_id}. Skipping BBox calculation.")
             return None

        x = float(row[X_COL])
        y = float(row[Y_COL])
        z = float(row[Z_COL])
        width = float(row[WIDTH_COL])
        depth = float(row[DEPTH_COL])
        height = float(row[HEIGHT_COL])

        # Basic validation for dimensions (optional but good practice)
        if width < 0 or depth < 0 or height < 0:
            logging.warning(f"Negative dimension(s) for element {global_id} (W:{width}, D:{depth}, H:{height}). Using absolute values for BBox.")
            width, depth, height = abs(width), abs(depth), abs(height)
        elif width == 0 or depth == 0 or height == 0:
             logging.warning(f"Zero dimension(s) for element {global_id} (W:{width}, D:{depth}, H:{height}). BBox might be a plane or line.")
             # Allow zero dimensions for now, could filter if needed

    except (ValueError, TypeError) as e:
        logging.error(f"Invalid non-numeric geometry value for element {global_id}: {e}. Row data: {row.to_dict()}")
        return None

    # Calculate bounding box corners
    bbox = {
        'min_x': x,
        'max_x': x + width,
        'min_y': y,
        'max_y': y + depth,
        'min_z': z,
        'max_z': z + height
    }
    return bbox

# --- Function to Check Intersection Between Two Bounding Boxes ---
def boxes_intersect(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
    """
    Checks whether two 3D Axis-Aligned Bounding Boxes (AABBs) intersect.

    Args:
        bbox1: The first bounding box dictionary.
        bbox2: The second bounding box dictionary.

    Returns:
        True if the boxes intersect, False otherwise.
    """
    # Check for separation along each axis (Separating Axis Theorem for AABBs)
    if bbox1['max_x'] <= bbox2['min_x'] or bbox1['min_x'] >= bbox2['max_x']:
        return False
    if bbox1['max_y'] <= bbox2['min_y'] or bbox1['min_y'] >= bbox2['max_y']:
        return False
    if bbox1['max_z'] <= bbox2['min_z'] or bbox1['min_z'] >= bbox2['max_z']:
        return False

    # If no separating axis is found, the boxes intersect
    return True

# --- Function to Detect Clashes ---
def detect_clashes(geometry_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects clashes between elements based on their bounding boxes.

    Args:
        geometry_df: DataFrame containing element geometry information.
                     Must include columns specified in GEOMETRY_COLUMNS.

    Returns:
        A list of dictionaries, where each dictionary represents a detected clash.
    """
    clashes: List[Dict[str, Any]] = []

    # Precompute bounding boxes for all valid elements
    bounding_boxes: Dict[str, Dict[str, float]] = {}
    valid_element_ids: List[str] = []

    logging.info("Computing bounding boxes for elements...")
    for index, row in geometry_df.iterrows():
        global_id = row.get(GLOBAL_ID_COL)
        if pd.isna(global_id):
            logging.warning(f"Skipping element at index {index} due to missing GlobalId.")
            continue

        global_id = str(global_id) # Ensure GlobalId is a string key
        bbox = compute_bounding_box(row)
        if bbox:
            if global_id in bounding_boxes:
                 logging.warning(f"Duplicate GlobalId '{global_id}' found. Overwriting bounding box. Check input data.")
            bounding_boxes[global_id] = bbox
            valid_element_ids.append(global_id)
        else:
            # Error/warning already logged in compute_bounding_box
            logging.debug(f"Failed to compute bounding box for element {global_id}.") # Debug level more appropriate here

    n = len(valid_element_ids)
    if n < 2:
        logging.info("Less than two elements with valid geometry found. No clash detection possible.")
        return clashes

    logging.info(f"Starting clash detection among {n} elements with valid geometry...")

    # Compare each pair of elements (naïve O(n^2) pairwise comparison)
    # For very large datasets, consider spatial indexing (e.g., Octree, k-d tree)
    # for faster candidate pair selection.
    clash_pairs = set() # Use a set to avoid duplicate reporting (Element1, Element2) vs (Element2, Element1)

    for i in range(n):
        id1 = valid_element_ids[i]
        bbox1 = bounding_boxes[id1]
        for j in range(i + 1, n):
            id2 = valid_element_ids[j]
            bbox2 = bounding_boxes[id2]

            # Create a unique sorted tuple key for the pair
            pair_key = tuple(sorted((id1, id2)))

            if pair_key not in clash_pairs:
                if boxes_intersect(bbox1, bbox2):
                    clash_detail = {
                        'Element1': id1,
                        'Element2': id2,
                        'ClashType': 'Bounding Box Intersection',
                        'Message': f"Bounding box overlap detected between elements {id1} and {id2}."
                    }
                    clashes.append(clash_detail)
                    clash_pairs.add(pair_key) # Mark pair as reported
                    # Log only significant clashes or sample them if too many
                    if len(clashes) <= 50: # Limit initial logging verbosity
                        logging.info(f"Clash detected: {id1} <-> {id2}")
                    elif len(clashes) == 51:
                        logging.info("Further clash detection logs will be summarized.")

    total_clashes = len(clashes)
    logging.info(f"Clash detection completed. Total clashes found: {total_clashes}")
    if total_clashes > 50:
         logging.info(f"({total_clashes - 50} additional clashes not logged individually)")
    return clashes

# --- Function to Save Clash Results ---
def save_clash_results_to_csv(clash_list: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves the list of clash dictionaries to a CSV file.

    Args:
        clash_list: The list of detected clashes.
        output_path: The file path where the CSV should be saved.
    """
    if not clash_list:
        logging.info("No clashes detected, so no results CSV file will be generated.")
        # Optionally create an empty file or a file with headers
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                logging.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)
            # Create empty file with headers
            pd.DataFrame(columns=['Element1', 'Element2', 'ClashType', 'Message']).to_csv(
                 output_path, index=False, encoding='utf-8'
            )
            logging.info(f"Empty clash results file created at: {output_path}")
        except Exception as e:
             logging.error(f"Could not create empty results file at {output_path}: {e}")
        return

    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        clash_df = pd.DataFrame(clash_list)
        clash_df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Clash detection results saved successfully to: {output_path}")

    except IOError as e:
        logging.error(f"Error writing clash results CSV file at {output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving clash results: {e}")

# --- Function to Print Clash Summary ---
def print_clash_summary(clash_list: List[Dict[str, Any]]) -> None:
    """Prints a formatted summary of the clash detection results to the console."""
    print("\n" + "="*70)
    print("              BIM-CAI Clash Detection Summary")
    print("="*70)
    if clash_list:
        count = len(clash_list)
        print(f"Total Clashes Detected: {count}")
        print("-"*70)
        # Print first few clashes for a quick overview
        max_print = 20
        for i, clash in enumerate(clash_list):
            if i < max_print:
                print(f"- Clash {i+1:03d}: {clash['Element1']} <-> {clash['Element2']} ({clash['ClashType']})")
            elif i == max_print:
                print(f"... and {count - max_print} more clashes. See CSV for full details.")
                break
        if count == 0: # Should be caught by the outer 'if' but good for safety
             print("No clashes were found.")
    else:
        print("✅ No bounding box clashes detected between elements.")
        print("Model appears clear based on this analysis.")
    print("="*70 + "\n")

# --- Main Execution Logic ---
def main() -> None:
    """Main function to run the clash detection process."""
    PROJECT_ROOT = get_project_root()

    # --- Define File Paths ---
    geometry_filename = "building_geometry_simple.csv"
    clash_results_filename = "clash_detection_results.csv" # More descriptive name

    geometry_file_path = os.path.join(PROJECT_ROOT, DATA_DIR, geometry_filename)
    output_file_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, clash_results_filename) # Use OUTPUT_DIR

    logging.info("--- Starting Clash Detection ---")
    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"Geometry Input: {geometry_file_path}")
    logging.info(f"Clash Results Output: {output_file_path}")

    # 1. Load Geometry Data
    geometry_df = load_csv_data(geometry_file_path, "Building Geometry")
    if geometry_df is None:
        logging.critical("Exiting: Failed to load the building geometry file.")
        print("❌ Error: Could not load the building geometry file. Check logs and file path.")
        return # Exit if loading failed

    # 2. Validate Data Format (Basic Check)
    if geometry_df.empty:
        logging.warning("Geometry file loaded but is empty. No elements to process.")
        print("⚠️ Geometry file is empty. Cannot perform clash detection.")
        # Create empty results file and exit cleanly
        save_clash_results_to_csv([], output_file_path)
        print_clash_summary([])
        logging.info("--- Clash Detection Finished (No Data) ---")
        return

    missing_cols = [col for col in GEOMETRY_COLUMNS if col not in geometry_df.columns]
    if missing_cols:
        logging.critical(f"Exiting: Missing required geometry columns in the input CSV: {missing_cols}")
        print(f"❌ Error: Input CSV is missing required columns: {missing_cols}. Cannot proceed.")
        return # Exit if columns missing

    # 3. Detect Clashes
    # Ensure GlobalId is treated as string to prevent potential float issues if IDs are numeric
    geometry_df[GLOBAL_ID_COL] = geometry_df[GLOBAL_ID_COL].astype(str)
    clashes = detect_clashes(geometry_df)

    # 4. Save Clash Results to CSV
    save_clash_results_to_csv(clashes, output_file_path)
    if clashes:
        print(f"\n✅ Clash detection complete. Results saved to {output_file_path}")
    else:
        print(f"\n✅ Clash detection complete. No clashes found. Empty results file saved to {output_file_path}")

    # 5. Print Clash Summary to Console
    print_clash_summary(clashes)
    logging.info("--- Clash Detection Finished ---")

if __name__ == "__main__":
    main()