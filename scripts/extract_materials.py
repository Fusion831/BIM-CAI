import pandas as pd
import os
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Function: Get Project Root ---
def get_project_root():
    """Gets the project root directory based on the script's location."""
    script_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(script_dir, '..'))

# --- Function to Extract Unique Materials ---
def extract_unique_materials(file_path):
    """Reads the metadata CSV and extracts unique material names."""
    unique_materials = []
    try:
        if not os.path.exists(file_path):
            logging.error(f"Metadata file not found at: {file_path}")
            return None # Indicate file not found

        # Read the CSV
        df = pd.read_csv(file_path)
        logging.info(f"Metadata file loaded. Found {len(df)} rows.")

        # Check if 'Material' column exists
        if 'Material' not in df.columns:
            logging.error("The 'Material' column was not found in the CSV file.")
            return [] # Return empty list, as column is missing

        # Get unique values from the 'Material' column
        # Use .dropna() to remove potential NaN values before unique()
        # Use .astype(str) to ensure all are treated as strings before comparison
        unique_materials = df['Material'].dropna().astype(str).unique()

        # Filter out the 'N/A' placeholder and sort alphabetically
        unique_materials = sorted([mat for mat in unique_materials if mat != 'N/A'])

        logging.info(f"Found {len(unique_materials)} unique material name(s) (excluding 'N/A').")
        return unique_materials

    except pd.errors.EmptyDataError:
        logging.error(f"Metadata file is empty: {file_path}")
        return [] # Return empty list for empty file
    except Exception as e:
        logging.error(f"An error occurred while processing {file_path}: {e}")
        return None # Indicate a general error occurred

# --- Main Execution Logic ---
def main():
    PROJECT_ROOT = get_project_root()
    metadata_filename = "building_metadata_simple.csv"
    metadata_file_path = os.path.join(PROJECT_ROOT, "data", metadata_filename)

    print(f"\nAttempting to extract unique materials from: {metadata_file_path}\n")

    materials = extract_unique_materials(metadata_file_path)

    if materials is None:
        print("❌ Extraction failed. Please check error messages above.")
    elif not materials: # Check if the list is empty
        print("ℹ️ No unique material names found (or only 'N/A' was present).")
    else:
        print("✅ Unique Material Names Found:")
        print("-" * 30)
        for material in materials:
            print(f"- {material}")
        print("-" * 30)
        print("\nUse this list to update your 'data/cost_database_simple.csv' file.")

if __name__ == "__main__":
    main()