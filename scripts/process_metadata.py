import ifcopenshell
import csv
import logging
import os

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function: Get Project Root ---
def get_project_root():
    """Gets the project root directory based on the script's location."""
    script_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(script_dir, '..'))

# --- Load IFC Function (Keep simple) ---
def load_ifc_file(file_path):
    """Load the IFC file from the given path."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"IFC file not found: {file_path}")
            return None
        ifc_file = ifcopenshell.open(file_path)
        logging.info(f"IFC file loaded successfully: {os.path.basename(file_path)}")
        return ifc_file
    except Exception as e:
        logging.error(f"Error loading IFC file {file_path}: {e}")
        return None

# --- Extract Project Info (Keep simple) ---
def extract_project_info(ifc_file):
    """Extract basic project name from the IFC file."""
    project_name = "Unknown Project"
    try:
        project = ifc_file.by_type("IfcProject")
        if project:
            project_name = project[0].Name if project[0].Name else project_name
        logging.info(f"Project Name: {project_name}")
        return project_name
    except Exception as e:
        logging.warning(f"Could not extract project info: {e}")
        return project_name

# --- Simplified Metadata Extraction ---
def extract_simplified_metadata(ifc_file):
    """
    Extracts simplified metadata (ID, Type, Name, a Material, basic Quantity)
    for key building elements. Prioritizes Area for Walls/Slabs.
    """
    # Define which element types to look for
    # Added IfcWallStandardCase as it's common
    element_types_to_extract = [
    "IfcWall", "IfcWallStandardCase",
    "IfcSlab",
    "IfcWindow",
    "IfcDoor",
    # "IfcPipeSegment",      # Incorrect for IFC2x3
    # "IfcDuctSegment",     # Incorrect for IFC2x3
    "IfcFlowSegment",       # CORRECT type for pipes/ducts in IFC2x3
    "IfcFlowTerminal",      # Exists in IFC2x3
]

    extracted_data = []
    processed_elements = 0
    processed_guids = set() # Avoid processing duplicates (e.g., IfcWall and IfcWallStandardCase)


    logging.info("Starting simplified metadata extraction...")

    for ifc_type in element_types_to_extract:
        elements = ifc_file.by_type(ifc_type)
        logging.info(f"Processing {len(elements)} elements of type {ifc_type}")

        for element in elements:
            guid = getattr(element, 'GlobalId', None)
            # Skip if no GUID or already processed
            if not guid or guid in processed_guids:
                continue
            processed_guids.add(guid)

            # Basic Info
            name = getattr(element, 'Name', 'N/A')
            tag = getattr(element, 'Tag', 'N/A')
            actual_ifc_type = element.is_a() # Get the most specific type

            # Placeholder for simple material and quantity
            material_name = 'N/A'
            quantity_value = 'N/A'
            quantity_name = 'N/A' # What quantity did we find? (e.g., Length, Area)

            # --- Attempt to find ONE associated material ---
            try:
                for assoc in getattr(element, 'HasAssociations', []):
                    if assoc.is_a('IfcRelAssociatesMaterial'):
                        mat_select = assoc.RelatingMaterial
                        if not mat_select: continue # Skip if no material defined

                        if mat_select.is_a('IfcMaterial'):
                            material_name = mat_select.Name or 'Unnamed Material'
                            break
                        elif mat_select.is_a('IfcMaterialLayerSetUsage'):
                            layer_set = mat_select.ForLayerSet
                            if layer_set and layer_set.MaterialLayers:
                                first_layer_mat = layer_set.MaterialLayers[0].Material
                                if first_layer_mat:
                                    material_name = first_layer_mat.Name or 'Unnamed Layer Material'
                                    break
                        elif mat_select.is_a('IfcMaterialList'):
                             mat = mat_select.Materials[0] if mat_select.Materials else None
                             if mat:
                                 material_name = mat.Name or 'Unnamed Material in List'
                                 break
                # Removed debug log here, enable if needed

            except Exception as e:
                logging.debug(f"Could not extract material for {guid}: {e}")


            # --- Attempt to find ONE common quantity (Area for Wall/Slab, else Length) ---
            quantity_found = False # Flag to stop searching definitions once quantity found
            try:
                for definition in getattr(element, 'IsDefinedBy', []):
                    if quantity_found: break # Stop searching definitions if found

                    if definition.is_a('IfcRelDefinesByProperties'):
                        prop_set_def = definition.RelatingPropertyDefinition
                        # Look specifically for BaseQuantities (very common)
                        # Also check common specific Qto sets like Qto_WallBaseQuantities
                        is_base_qto = prop_set_def and prop_set_def.is_a('IfcElementQuantity')
                        if is_base_qto and ('BaseQuantities' in prop_set_def.Name or 'BaseQto' in prop_set_def.Name):
                            is_wall_or_slab = actual_ifc_type in ["IfcWall", "IfcWallStandardCase", "IfcSlab"]

                            # --- Loop through individual quantities within the set ---
                            for quantity in prop_set_def.Quantities:
                                # 1. Prioritize Area for Walls/Slabs
                                if is_wall_or_slab:
                                    if quantity.is_a('IfcQuantityArea') and quantity.Name in ['Area', 'NetArea', 'GrossArea']:
                                        quantity_name = quantity.Name # Use the actual name found
                                        quantity_value = quantity.AreaValue
                                        quantity_found = True
                                        logging.debug(f"Found Area '{quantity_name}'={quantity_value} for {guid}")
                                        break # Found preferred quantity (Area), stop inner loop

                                # 2. If not Wall/Slab, or if Area wasn't found, look for Length
                                if not quantity_found: # Only look for length if area wasn't the target/found
                                     if quantity.is_a('IfcQuantityLength') and quantity.Name == 'Length':
                                        quantity_name = 'Length'
                                        quantity_value = quantity.LengthValue
                                        quantity_found = True
                                        logging.debug(f"Found Length={quantity_value} for {guid}")
                                        break # Found a quantity, stop inner loop

                            # 3. If we found a quantity in this Qto set, stop searching other definitions
                            if quantity_found:
                                break # Stop outer loop (definitions)

            except Exception as e:
                 logging.debug(f"Could not extract quantity for {guid}: {e}")


            # Add the collected data as a dictionary
            extracted_data.append({
                'GlobalId': guid,
                'IfcType': actual_ifc_type, # Store the specific type found
                'Name': name,
                'Tag': tag,
                'Material': material_name,
                'QuantityName': quantity_name,
                'QuantityValue': quantity_value
            })
            processed_elements += 1

    logging.info(f"Finished simplified extraction. Processed {processed_elements} elements.")
    return extracted_data

# --- Write Simplified CSV ---
def write_simplified_metadata_to_csv(data, csv_file):
    """Writes the simplified extracted data to a CSV file."""
    if not data:
        logging.warning("No element data extracted, CSV file will not be created/updated.")
        # Optional: Create empty file with header if preferred
        # header = ['GlobalId', 'IfcType', 'Name', 'Tag', 'Material', 'QuantityName', 'QuantityValue']
        # with open(csv_file, mode="w", newline="", encoding='utf-8') as file:
        #     writer = csv.DictWriter(file, fieldnames=header)
        #     writer.writeheader()
        return

    # Define the exact columns we want in our CSV
    header = ['GlobalId', 'IfcType', 'Name', 'Tag', 'Material', 'QuantityName', 'QuantityValue']

    try:
        output_dir = os.path.dirname(csv_file)
        if not os.path.exists(output_dir):
            logging.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        with open(csv_file, mode="w", newline="", encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header, restval='', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        logging.info(f"Simplified metadata written successfully to: {csv_file}")

    except IOError as e:
        logging.error(f"Error writing CSV file at {csv_file}: {e} (Check permissions or path)")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing CSV: {e}")

# --- Main Execution Logic ---
def main():
    PROJECT_ROOT = get_project_root()
    # !!! IMPORTANT: Change this to your actual IFC file name !!!
    ifc_filename = "Your_New_Architectural_Model_2.ifc" # Make sure this file exists
    ifc_file_path = os.path.join(PROJECT_ROOT, "models", ifc_filename)

    # Output file name
    csv_filename = "building_metadata_simple.csv"
    csv_output_path = os.path.join(PROJECT_ROOT, "data", csv_filename)

    logging.info(f"--- Starting Simplified Metadata Extraction (Prioritizing Area for Walls/Slabs) ---")
    logging.info(f"Input IFC File: {ifc_file_path}")
    logging.info(f"Output CSV File: {csv_output_path}")

    # 1. Load IFC
    ifc_file = load_ifc_file(ifc_file_path)
    if not ifc_file:
        logging.critical("Exiting: Cannot proceed without IFC file.")
        return

    # 2. Extract Project Info
    extract_project_info(ifc_file)

    # 3. Extract Simplified Metadata (with updated logic)
    metadata = extract_simplified_metadata(ifc_file)

    # 4. Write to CSV
    if metadata: # Check if the list is not empty
        write_simplified_metadata_to_csv(metadata, csv_output_path)
        print(f"✅ Simplified metadata extraction complete! {len(metadata)} elements saved to {csv_output_path}")
    else:
        logging.warning("No metadata was extracted or processed.")
        print(f"⚠️ Simplified metadata extraction finished, but no data was generated or saved.")

    logging.info(f"--- Simplified Metadata Extraction Finished ---")


if __name__ == "__main__":
    main()