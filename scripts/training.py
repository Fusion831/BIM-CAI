import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler # Example scaler
from sklearn.impute import SimpleImputer # Example imputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Constants ---
TRAINING_DATA_FILE = "training_data_features.csv" #  aggregated features + targets
DATA_DIR = "data"
MODEL_DIR = "ml_models" # Directory to save models

# --- Feature and Target Columns ---
FEATURE_COLUMNS = [
    'TotalFloorArea',
    'TotalExtWallArea',
    'TotalWindowArea',
    'WindowToWallRatio',
    'BuildingVolume',
    # Add ALL other feature column names used for training
]

TARGET_ENERGY_COL = 'Actual_Energy_Efficiency' #  ground truth energy column name
TARGET_CARBON_COL = 'Actual_Carbon_Footprint' # ground truth carbon column name

# Files to be saved
ENERGY_MODEL_FILENAME = "energy_model.joblib"
CARBON_MODEL_FILENAME = "carbon_model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Get Project Root ---
def get_project_root() -> str:
    try:
        script_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(script_dir, '..'))
    except NameError:
        logging.warning("_file_ not defined. Assuming CWD is project root.")
        return os.path.abspath('.')

# --- Main Training Logic ---
def main():
    PROJECT_ROOT = get_project_root()
    data_path = os.path.join(PROJECT_ROOT, DATA_DIR, TRAINING_DATA_FILE)
    model_output_dir = os.path.join(PROJECT_ROOT, MODEL_DIR)

    # Create model directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    logging.info(f"Model output directory: {model_output_dir}")

    # 1. Load Data
    logging.info(f"Loading training data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
        if df.empty:
             logging.error("Training data file is empty.")
             return
        # Basic validation
        required_cols = FEATURE_COLUMNS + [TARGET_ENERGY_COL, TARGET_CARBON_COL]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logging.error(f"Training data missing required columns: {missing}")
            return
        logging.info(f"Loaded {len(df)} training samples.")
    except FileNotFoundError:
        logging.error(f"Training data file not found: {data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading training data: {e}", exc_info=True)
        return

    # 2. Prepare Features (X) and Targets (y)
    X = df[FEATURE_COLUMNS]
    y_energy = df[TARGET_ENERGY_COL]
    y_carbon = df[TARGET_CARBON_COL]

    #Handling division by zero or other stuff
    X = X.replace([np.inf, -np.inf], np.nan) # Replace infinities with NaN for imputation

    # 3. Split Data (Train/Test)
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_energy_train, y_energy_test, y_carbon_train, y_carbon_test = train_test_split(
        X, y_energy, y_carbon, test_size=0.2, random_state=42 # Use random_state for reproducibility
    )
    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 4. Define Preprocessing Pipeline
    #    - Impute missing values (e.g., using the mean or median)
    #    - Scale numerical features (important for many models, including RF to some extent)
    logging.info("Defining preprocessing pipeline...")
    # Assuming all features are numeric for this 
    numeric_features = FEATURE_COLUMNS 
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle NaNs
        ('scaler', StandardScaler())
    ])

    # Create the full preprocessor
    #Assuming no categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
            # ('cat', categorical_transformer, categorical_features) # Example
        ],
        remainder='passthrough' # Keep other columns if any; 'drop' is safer if only features expected
    )

    # 5. Define Models (Random Forest Regressor)
    # You might want to tune hyperparameters (n_estimators, max_depth, etc.) using GridSearchCV or RandomizedSearchCV
    energy_model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all cores
    carbon_model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # 6. Create Full Pipelines (Preprocessor + Model)
    # This ensures preprocessing is applied consistently during training, prediction, and evaluation
    energy_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', energy_model_rf)])

    carbon_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', carbon_model_rf)])

    # 7. Train Models
    logging.info("Training Energy Prediction Model...")
    energy_pipeline.fit(X_train, y_energy_train)
    logging.info("Energy Model Training Complete.")

    logging.info("Training Carbon Prediction Model...")
    carbon_pipeline.fit(X_train, y_carbon_train)
    logging.info("Carbon Model Training Complete.")

    # 8. Evaluate Models on Test Set
    logging.info("Evaluating models on the test set...")

    # Energy Model Evaluation
    y_energy_pred = energy_pipeline.predict(X_test)
    energy_mae = mean_absolute_error(y_energy_test, y_energy_pred)
    energy_rmse = np.sqrt(mean_squared_error(y_energy_test, y_energy_pred))
    energy_r2 = r2_score(y_energy_test, y_energy_pred)
    logging.info("--- Energy Model Performance ---")
    logging.info(f"  MAE: {energy_mae:.4f}")
    logging.info(f"  RMSE: {energy_rmse:.4f}")
    logging.info(f"  R²: {energy_r2:.4f}")
    print(f"\n--- Energy Model Performance ---")
    print(f"  MAE:  {energy_mae:.4f}")
    print(f"  RMSE: {energy_rmse:.4f}")
    print(f"  R²:   {energy_r2:.4f}")


    # Carbon Model Evaluation
    y_carbon_pred = carbon_pipeline.predict(X_test)
    carbon_mae = mean_absolute_error(y_carbon_test, y_carbon_pred)
    carbon_rmse = np.sqrt(mean_squared_error(y_carbon_test, y_carbon_pred))
    carbon_r2 = r2_score(y_carbon_test, y_carbon_pred)
    logging.info("--- Carbon Model Performance ---")
    logging.info(f"  MAE: {carbon_mae:.4f}")
    logging.info(f"  RMSE: {carbon_rmse:.4f}")
    logging.info(f"  R²: {carbon_r2:.4f}")
    print(f"\n--- Carbon Model Performance ---")
    print(f"  MAE:  {carbon_mae:.4f}")
    print(f"  RMSE: {carbon_rmse:.4f}")
    print(f"  R²:   {carbon_r2:.4f}")

    # 9. Save the Necessary Assets
    #    - The fitted preprocessor (contains imputer state, scaler means/stds)
    #    - The trained energy model
    #    - The trained carbon model

    # IMPORTANT: Save the FITTED preprocessor separately
    # The predict_environment.py script will load it separately
    preprocessor_path = os.path.join(model_output_dir, PREPROCESSOR_FILENAME)
    joblib.dump(energy_pipeline.named_steps['preprocessor'], preprocessor_path)
    logging.info(f"Preprocessor saved to: {preprocessor_path}")
    print(f"\n✅ Preprocessor saved to: {preprocessor_path}")


    # Save the models (which were trained within the pipeline)
    #Save just the fitted regressor (matches current predict_environment.py structure)

    
    energy_model_path = os.path.join(model_output_dir, ENERGY_MODEL_FILENAME)
    joblib.dump(energy_pipeline.named_steps['regressor'], energy_model_path)
    logging.info(f"Energy model saved to: {energy_model_path}")
    print(f"✅ Energy model saved to: {energy_model_path}")


    carbon_model_path = os.path.join(model_output_dir, CARBON_MODEL_FILENAME)
    joblib.dump(carbon_pipeline.named_steps['regressor'], carbon_model_path)
    logging.info(f"Carbon model saved to: {carbon_model_path}")
    print(f"✅ Carbon model saved to: {carbon_model_path}")


    logging.info("--- Training Process Finished ---")

if __name__ == "_main_":
    main()