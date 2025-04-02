import pandas as pd
import os
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Get Project Root ---
def get_project_root():
    """Gets the project root directory based on the script's location."""
    script_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(script_dir, '..'))

# --- Function to Load Data using Pandas ---
def load_csv_data(file_path, file_description):
    """Loads data from a CSV file using pandas."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"{file_description} file not found at: {file_path}")
            return None
        df = pd.read_csv(file_path)
        logging.info(f"{file_description} loaded successfully. Found {len(df)} rows.")
        # Handle potential NaN values (empty cells) that pandas reads
        df = df.fillna('N/A') # Replace empty cells with 'N/A' string for consistency
        return df
    except pd.errors.EmptyDataError:
         logging.error(f"Error loading {file_description} file {file_path}: File is empty or has no columns.")
         return None # Return None specifically for empty file case after logging
    except Exception as e:
        logging.error(f"Error loading {file_description} file {file_path}: {e}")
        return None

# --- Function to Calculate Costs and Material Usage ---
def calculate_costs_and_usage(metadata_df, cost_db_df):
    """
    Calculates estimated cost per element, total cost,
    total quantity used per material/unit, and total cost per material.
    """
    results = []
    total_cost = 0
    material_quantities = {} # Key: (Material, Unit), Value: Total Quantity
    material_costs = {}      # Key: Material, Value: Total Cost

    logging.info("Starting cost and material usage calculation...")

    # --- Prepare Cost Lookup ---
    cost_lookup = {}
    if cost_db_df is not None: # Check if cost_db loaded successfully
        for _, row in cost_db_df.iterrows():
             # Ensure UnitCost is treated as float, handle potential errors
            try:
                unit_cost_float = float(row['UnitCost'])
                cost_lookup[(row['IfcType'], row['Material'])] = {'Unit': row['Unit'], 'UnitCost': unit_cost_float}
            except ValueError:
                 logging.warning(f"Could not convert UnitCost '{row['UnitCost']}' to float for {row['IfcType']}/{row['Material']}. Skipping this entry.")
    else:
        logging.error("Cost Database is missing or empty. Cannot perform cost lookup.")
        # Return empty results if cost db is missing
        return pd.DataFrame(results), total_cost, material_quantities, material_costs


    # --- Iterate through Metadata Elements ---
    for index, element in metadata_df.iterrows():
        element_type = element['IfcType']
        element_material = element['Material'] # Might be 'N/A'
        quantity_name = element['QuantityName'] # Might be 'N/A', 'Length', 'Area'
        quantity_value = element['QuantityValue'] # Might be 'N/A' or a number

        unit_cost = 0
        cost_unit = 'N/A'
        element_cost = 0
        cost_basis = "No Match in Cost DB" # Default reason if no cost found
        q_val_float = 0 # Initialize quantity as float

        # --- Find Cost Info ---
        cost_info = cost_lookup.get((element_type, element_material))
        if not cost_info: # Try generic fallback
            cost_info = cost_lookup.get((element_type, 'Generic'))

        # --- Calculate Cost and Aggregate Usage if Match Found ---
        if cost_info:
            unit_cost = cost_info['UnitCost']
            cost_unit = cost_info['Unit']

            try:
                # Try converting quantity value to float
                q_val_float = float(quantity_value) if quantity_value != 'N/A' else 0

                # --- Determine Cost Based on Unit ---
                if cost_unit == 'm2' and quantity_name == 'Area' and q_val_float > 0:
                    element_cost = q_val_float * unit_cost
                    cost_basis = f"Area ({q_val_float:.2f} m2) * UnitCost ({unit_cost})"
                    # Aggregate quantity
                    key = (element_material, 'm2')
                    material_quantities[key] = material_quantities.get(key, 0) + q_val_float

                elif cost_unit == 'm' and quantity_name == 'Length' and q_val_float > 0:
                    element_cost = q_val_float * unit_cost
                    cost_basis = f"Length ({q_val_float:.2f} m) * UnitCost ({unit_cost})"
                     # Aggregate quantity
                    key = (element_material, 'm')
                    material_quantities[key] = material_quantities.get(key, 0) + q_val_float

                elif cost_unit == 'Item':
                    element_cost = unit_cost
                    cost_basis = f"Per Item Cost ({unit_cost})"
                     # Aggregate quantity (count items)
                    key = (element_material, 'Items') # Use 'Items' as unit
                    material_quantities[key] = material_quantities.get(key, 0) + 1

                else: # Mismatch between cost unit and available quantity name/value
                    if cost_unit == 'Item': # Fallback to item cost if quantity mismatch
                         element_cost = unit_cost
                         cost_basis = f"Per Item Cost ({unit_cost}) - Quantity N/A or Mismatch"
                         key = (element_material, 'Items')
                         material_quantities[key] = material_quantities.get(key, 0) + 1
                    else:
                        cost_basis = "Unit Cost Found - No Matching Quantity"
                        element_cost = 0

                # --- Aggregate Costs per Material ---
                # Use the material name that was *used* for the lookup (could be 'Generic')
                material_used_for_costing = element_material if cost_lookup.get((element_type, element_material)) else 'Generic'
                # Only add cost if element_cost > 0 (or adjust if you want to track materials with 0 cost)
                if element_cost > 0:
                    material_costs[material_used_for_costing] = material_costs.get(material_used_for_costing, 0) + element_cost

            except ValueError:
                 cost_basis = "Unit Cost Found - Invalid Quantity Value"
                 logging.warning(f"Element {element['GlobalId']} ({element_type}): Invalid number format for QuantityValue: '{quantity_value}'. Assigning cost 0.")
                 element_cost = 0
            except Exception as e:
                 cost_basis = "Error during calculation"
                 logging.error(f"Calculation error for element {element['GlobalId']}: {e}")
                 element_cost = 0
        else:
            # No match found in cost database
            element_cost = 0
            # cost_basis remains "No Match in Cost DB"

        # --- Append Detailed Results for CSV ---
        results.append({
            'GlobalId': element['GlobalId'],
            'IfcType': element_type,
            'Name': element['Name'],
            'Material': element_material, # Original material from metadata
            'QuantityName': quantity_name,
            'QuantityValue': quantity_value,
            'MatchedUnitCost': unit_cost if cost_info else 0,
            'CostUnit': cost_unit,
            'CalculatedElementCost': round(element_cost, 2),
            'CostBasis': cost_basis
        })
        total_cost += element_cost # Accumulate total cost

    logging.info(f"Finished cost calculations. Total Estimated Cost: {total_cost:.2f}")
    return pd.DataFrame(results), total_cost, material_quantities, material_costs

# --- Function to Print Formatted Summary ---
def print_summary(total_cost, material_quantities, material_costs):
    """Prints a formatted summary of the cost analysis to the console."""

    print("\n" + "="*60)
    print("    BIM-CAI Cost Analysis Summary")
    print("="*60)

    # --- Material Quantity Summary ---
    print("\n--- Material Quantity Summary ---")
    if material_quantities:
        # Sort by Material, then Unit for consistent output
        sorted_mat_qtys = sorted(material_quantities.items())
        print(f"{'Material':<25} {'Unit':<10} {'Total Quantity':>15}")
        print("-"*60)
        for (material, unit), quantity in sorted_mat_qtys:
            print(f"{material:<25} {unit:<10} {quantity:>15.2f}")
    else:
        print("No material quantities were aggregated.")

    # --- Material Cost Summary ---
    print("\n--- Material Cost Summary ---")
    if material_costs:
         # Sort by Material name
        sorted_mat_costs = sorted(material_costs.items())
        print(f"{'Material':<35} {'Total Cost':>15}")
        print("-"*60)
        for material, cost in sorted_mat_costs:
            print(f"{material:<35} {cost:>15.2f}")
    else:
        print("No costs were aggregated by material.")

    # --- Overall Total Cost ---
    print("\n" + "-"*60)
    print(f"{'>>> TOTAL ESTIMATED PROJECT COST:':<45} {total_cost:>15.2f}")
    print("="*60 + "\n")


# --- Function to Save Results (Keep As Is) ---
def save_results_to_csv(results_df, output_path):
    """Saves the results DataFrame to a CSV file."""
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        results_df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Cost analysis results saved successfully to: {output_path}")
    except IOError as e:
        logging.error(f"Error writing results CSV file at {output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing results CSV: {e}")


# --- Main Execution Logic ---
def main():
    PROJECT_ROOT = get_project_root()

    # --- Define File Paths ---
    metadata_filename = "building_metadata_simple.csv"
    metadata_file_path = os.path.join(PROJECT_ROOT, "data", metadata_filename)

    cost_db_filename = "cost_database_simple.csv"
    cost_db_file_path = os.path.join(PROJECT_ROOT, "data", cost_db_filename)

    output_filename = "cost_analysis_simple.csv"
    output_file_path = os.path.join(PROJECT_ROOT, "data", output_filename)

    logging.info(f"--- Starting Simple Cost Analysis & Usage Aggregation ---")
    logging.info(f"Metadata Input: {metadata_file_path}")
    logging.info(f"Cost DB Input: {cost_db_file_path}")
    logging.info(f"Output File: {output_file_path}")

    # 1. Load Input Data
    metadata_df = load_csv_data(metadata_file_path, "Building Metadata")
    cost_db_df = load_csv_data(cost_db_file_path, "Cost Database")

    # Check if data loading was successful
    if metadata_df is None or cost_db_df is None:
        logging.critical("Exiting: Failed to load required input files.")
        # Attempt to provide more specific feedback if one file loaded but not the other
        if metadata_df is None:
             print("❌ Error: Could not load the building metadata file. Please check logs.")
        if cost_db_df is None:
             print("❌ Error: Could not load the cost database file. Please check logs.")
        return

    # Handle case where metadata is empty
    if metadata_df.empty:
        logging.warning("Metadata file is empty. No elements to analyze.")
        print("⚠️ Metadata file is empty. Cannot perform cost analysis.")
        return

    # 2. Calculate Costs and Aggregate Usage
    cost_results_df, total_estimated_cost, mat_qtys, mat_costs = calculate_costs_and_usage(metadata_df, cost_db_df)

    # 3. Save Detailed Results to CSV
    if not cost_results_df.empty:
        save_results_to_csv(cost_results_df, output_file_path)
        print(f"\n✅ Detailed cost analysis per element saved to {output_file_path}")
    else:
        # This case might happen if metadata loaded but no costs could be calculated
        logging.warning("Cost analysis finished, but no results rows were generated for the CSV.")
        print(f"⚠️ Detailed cost analysis per element generated no results rows.")


    # 4. Print Formatted Summary to Console
    print_summary(total_estimated_cost, mat_qtys, mat_costs)

    logging.info(f"--- Simple Cost Analysis & Usage Aggregation Finished ---")


if __name__ == "__main__":
    main()