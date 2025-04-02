import pandas as pd
import joblib
import os
import logging
from typing import Optional, Dict, Any

# --- Constants ---
# File/Directory Names (Adjust as needed)
METADATA_FILENAME = "building_metadata.csv"
ENERGY_EFFICIENCY_FILENAME = "energy_efficiency.csv"
ENERGY_MODEL_FILENAME = "energy_model.joblib"
CARBON_MODEL_FILENAME = "carbon_model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

DATA_DIR = "data"
OUTPUT_DIR = "data" # Saving results back to data dir for now
MODEL_DIR = "ml_models" # Assumes models are stored here

# Assumed Columns from building_metadata.csv (TODO: Adjusting based on the actual Data(Template for the code))
ENTITY_TYPE_COL = 'IfcEntityType' # Or similar column name identifying the IFC type
AREA_COL = 'Area'
VOLUME_COL = 'Volume'
IS_EXTERNAL_COL = 'IsExternal' # Assuming boolean/numeric indicating if wall/slab is external
GLOBAL_ID_COL = 'GlobalId'

# Feature Names
#(TODO: Adjust these based on the actual feature engineering)
FEATURE_TOTAL_FLOOR_AREA = 'TotalFloorArea'
FEATURE_TOTAL_EXT_WALL_AREA = 'TotalExtWallArea'
FEATURE_TOTAL_WINDOW_AREA = 'TotalWindowArea'
FEATURE_WWR = 'WindowToWallRatio'
FEATURE_BUILDING_VOLUME = 'BuildingVolume'
#(TODO: ADD MORE)

EXPECTED_FEATURES = [
    FEATURE_TOTAL_FLOOR_AREA,
    FEATURE_TOTAL_EXT_WALL_AREA,
    FEATURE_TOTAL_WINDOW_AREA,
    FEATURE_WWR,
    FEATURE_BUILDING_VOLUME,
    #(TODO ADD HERE TOO)
]


# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Get Project Root (Same as other scripts) ---
def get_project_root() -> str:
    """Gets the project root directory based on the script's location."""
    try:
        script_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(script_dir, '..'))
    except NameError:
        logging.warning("_file_ not defined. Assuming current working directory is project root.")
        return os.path.abspath('.')

# --- Function to Load ML Models and Preprocessor ---
def load_prediction_assets(model_dir: str) -> Optional[Dict[str, Any]]:
    """TO load  the saved ML models and preprocessor."""
    assets = {}
    asset_files = {
        'energy_model': ENERGY_MODEL_FILENAME,
        'carbon_model': CARBON_MODEL_FILENAME,
        'preprocessor': PREPROCESSOR_FILENAME
    }
    try:
        for key, filename in asset_files.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                 logging.error(f"Required asset file not found: {path}")
                 return None
            assets[key] = joblib.load(path)
            logging.info(f"Successfully loaded {key} from {filename}")
        return assets
    except Exception as e:
        logging.error(f"Error loading prediction assets from {model_dir}: {e}", exc_info=True)
        return None

# --- Function to Load Building Metadata ---
def load_building_metadata(file_path: str) -> Optional[pd.DataFrame]:
    """Loads the building metadata CSV file."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"Metadata file not found: {file_path}")
            return None
        df = pd.read_csv(file_path)
        logging.info(f"Loaded building metadata from {os.path.basename(file_path)}. Found {len(df)} rows.")
        if df.empty:
            logging.warning(f"Metadata file {file_path} is empty.")
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"Error loading metadata file {file_path}: File is empty.")
        return None
    except Exception as e:
        logging.error(f"Error loading metadata file {file_path}: {e}", exc_info=True)
        return None

# --- Function to Extract and Engineer Features ---
#(TODO EDIT HERE AS WELL)
def extract_and_engineer_features(metadata_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extracts raw data from metadata and engineers the features required by the models.

    Args:
        metadata_df: DataFrame loaded from building_metadata.csv.

    Returns:
        A DataFrame with ONE row containing the engineered features,
        with columns matching EXPECTED_FEATURES, or None if errors occur.
    """
    logging.info("Starting feature extraction and engineering...")
    required_cols = [ENTITY_TYPE_COL, AREA_COL, VOLUME_COL] # Add others needed
    if not all(col in metadata_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in metadata_df.columns]
        logging.error(f"Metadata DataFrame missing required columns for feature engineering: {missing}")
        return None

    features = {}

    try:
        # TODO: Refine filter based on actual entity types used for floors
        floor_area = metadata_df[metadata_df[ENTITY_TYPE_COL] == 'IfcSlab'][AREA_COL].sum()
        features[FEATURE_TOTAL_FLOOR_AREA] = floor_area

        # TODO: Refine filter for walls (e.g., 'IfcWallStandardCase') and use IS_EXTERNAL_COL if available
        # Assuming IS_EXTERNAL_COL is 1 for external, 0 for internal
        if IS_EXTERNAL_COL in metadata_df.columns:
             ext_wall_area = metadata_df[
                 (metadata_df[ENTITY_TYPE_COL].str.contains('IfcWall', case=False, na=False)) &
                 (metadata_df[IS_EXTERNAL_COL] == 1) # Adjust condition if IsExternal is boolean or string
             ][AREA_COL].sum()
        else:
             logging.warning(f"Column '{IS_EXTERNAL_COL}' not found. Calculating total wall area instead.")
             ext_wall_area = metadata_df[
                 metadata_df[ENTITY_TYPE_COL].str.contains('IfcWall', case=False, na=False)
             ][AREA_COL].sum()
        features[FEATURE_TOTAL_EXT_WALL_AREA] = ext_wall_area

        # Example Feature 3: Total Window Area
        # TODO: Refine filter for windows (e.g., 'IfcWindow')
        window_area = metadata_df[metadata_df[ENTITY_TYPE_COL] == 'IfcWindow'][AREA_COL].sum()
        features[FEATURE_TOTAL_WINDOW_AREA] = window_area

        # Example Feature 4: Window-to-Wall Ratio (WWR)
        if ext_wall_area > 0:
            wwr = window_area / ext_wall_area
        else:
            wwr = 0
            logging.warning("External wall area is zero, setting WWR to 0.")
        features[FEATURE_WWR] = wwr
        # TODO: Refine filter for spaces ('IfcSpace')
        building_volume = metadata_df[metadata_df[ENTITY_TYPE_COL] == 'IfcSpace'][VOLUME_COL].sum()
        features[FEATURE_BUILDING_VOLUME] = building_volume

        # --- Add calculations for ALL other features required by your models ---
        # Example: counts of materials, specific U-values, etc.
        # Ensure the keys in the 'features' dict match EXPECTED_FEATURES

        # Convert features dict to a DataFrame with one row
        features_df = pd.DataFrame([features])

        # Ensure all expected columns are present, even if calculated as 0 or NaN initially
        for col in EXPECTED_FEATURES:
            if col not in features_df.columns:
                logging.warning(f"Expected feature '{col}' was not calculated. Adding column with NaN.")
                features_df[col] = pd.NA # Or 0, depending on how your model handles missing features

        # Reorder columns to match the order expected by the preprocessor/model
        features_df = features_df[EXPECTED_FEATURES]

        logging.info("Finished feature extraction and engineering.")
        logging.debug(f"Engineered features:\n{features_df.to_string()}")
        return features_df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return None


# --- Main Execution Logic ---
def main() -> None:
    """Main function to run the prediction process."""
    PROJECT_ROOT = get_project_root()

    # --- Define File Paths ---
    model_dir_path = os.path.join(PROJECT_ROOT, MODEL_DIR)
    metadata_file_path = os.path.join(PROJECT_ROOT, DATA_DIR, METADATA_FILENAME)
    output_file_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, ENERGY_EFFICIENCY_FILENAME)

    logging.info("--- Starting Environment Prediction ---")
    logging.info(f"Project Root: {PROJECT_ROOT}")
    logging.info(f"Model Directory: {model_dir_path}")
    logging.info(f"Input Metadata: {metadata_file_path}")
    logging.info(f"Output Predictions: {output_file_path}")

    # 1. Load ML Assets
    logging.info("Loading ML models and preprocessor...")
    ml_assets = load_prediction_assets(model_dir_path)
    if ml_assets is None:
        logging.critical("Failed to load ML assets. Exiting.")
        print("❌ Error: Could not load necessary machine learning model files. Check logs.")
        return

    # 2. Load Building Metadata
    logging.info("Loading building metadata...")
    metadata_df = load_building_metadata(metadata_file_path)
    if metadata_df is None or metadata_df.empty:
        logging.critical("Failed to load or empty building metadata. Cannot proceed.")
        print("❌ Error: Could not load or empty building metadata CSV. Check logs and file.")
        return

    # 3. Extract and Engineer Features
    logging.info("Extracting and engineering features from metadata...")
    features_df = extract_and_engineer_features(metadata_df)
    if features_df is None or features_df.empty:
        logging.critical("Failed to engineer features from metadata. Exiting.")
        print("❌ Error: Could not process metadata into features. Check logs.")
        return

    # 4. Preprocess Features
    # Use the loaded preprocessor's transform method ONLY (DO NOT FIT)
    preprocessor = ml_assets['preprocessor']
    try:
        logging.info("Preprocessing features using loaded preprocessor...")
        # Ensure features_df has the exact columns in the exact order the preprocessor expects
        processed_features = preprocessor.transform(features_df)
        logging.info("Features preprocessed successfully.")
        # Logging the shape might be useful for debugging
        logging.debug(f"Shape of processed features: {processed_features.shape}")
    except Exception as e:
        logging.error(f"Error applying preprocessor: {e}", exc_info=True)
        logging.error("Ensure feature columns match EXACTLY those used for training the preprocessor.")
        print("❌ Error: Failed to preprocess features. Check logs.")
        return

    # 5. Make Predictions
    energy_model = ml_assets['energy_model']
    carbon_model = ml_assets['carbon_model']
    try:
        logging.info("Making predictions...")
        predicted_energy = energy_model.predict(processed_features)
        predicted_carbon = carbon_model.predict(processed_features)
        logging.info("Predictions completed.")
        # Predictions are likely numpy arrays of length 1
        energy_val = predicted_energy[0]
        carbon_val = predicted_carbon[0]
        logging.info(f"Predicted Energy Efficiency: {energy_val:.2f} kWh/m²/yr (example unit)")
        logging.info(f"Predicted Carbon Footprint: {carbon_val:.2f} kg CO2e (example unit)")

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        print("❌ Error: Failed to make predictions. Check logs.")
        return

    # 6. Save Results
    try:
        logging.info(f"Saving prediction results to {output_file_path}...")
        # Add an identifier if multiple predictions might be stored, otherwise overwrite
        results_df = pd.DataFrame({
            'BuildingIdentifier': [os.path.basename(metadata_file_path).replace('.csv', '')], # Example ID
            'Predicted_Energy_Efficiency': [energy_val],
            'Predicted_Carbon_Footprint': [carbon_val],
            'Energy_Unit': ['kWh/m²/yr (assumed)'], # Clarify units
            'Carbon_Unit': ['kg CO2e (assumed)']    # Clarify units
        })
        results_df.to_csv(output_file_path, index=False, encoding='utf-8')
        logging.info("Prediction results saved successfully.")
        print(f"\n✅ Prediction results saved to {output_file_path}")

    except Exception as e:
        logging.error(f"Error saving prediction results: {e}", exc_info=True)
        print("❌ Error: Failed to save prediction results. Check logs.")

    logging.info("--- Environment Prediction Finished ---")


if __name__ == "_main_":
    main()