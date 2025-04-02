import streamlit as st
import ifcopenshell
import ifcopenshell.geom
import pandas as pd
import os
import tempfile
import json
import uuid # For unique component keys
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
st.set_page_config(layout="wide", page_title="BIM-CAI Dashboard")

# Define which IFC elements to extract geometry and metadata for
TARGET_IFC_ELEMENTS = [
    "IfcWall", "IfcWallStandardCase",
    "IfcSlab",
    "IfcBeam",
    "IfcColumn",
    "IfcWindow",
    "IfcDoor",
    "IfcStair", "IfcStairFlight",
    "IfcRoof",
    "IfcBuildingElementProxy", # Catch-all for some elements
    "IfcSpace", # Often useful for context, maybe style differently
    "IfcCovering",
    "IfcRailing",
    "IfcPlate",
    "IfcMember",
    # Add more as needed, e.g., MEP elements: "IfcPipeSegment", "IfcDuctSegment"
]

# --- ifcopenshell Geometry Settings ---
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)
settings.set(settings.WELD_VERTICES, True) # Try to merge duplicate vertices
settings.set(settings.STRICT_TOLERANCE, False)
settings.set(settings.INCLUDE_CURVES, True) # Generate meshes from curves


# --- Helper Functions ---

@st.cache_data(show_spinner=False) # Cache results based on file content
def get_material_info(element):
    """Extracts material information for an IFC element. Caching this helps."""
    materials = []
    try:
        # Common case: Association through IfcRelAssociatesMaterial
        for rel in element.HasAssociations:
            if rel.is_a("IfcRelAssociatesMaterial"):
                material_select = rel.RelatingMaterial
                # Handle different material assignment types more robustly
                if hasattr(material_select, 'ForLayerSet'): # IfcMaterialLayerSetUsage
                    layer_set = material_select.ForLayerSet
                    if hasattr(layer_set, 'MaterialLayers'):
                        for layer in layer_set.MaterialLayers:
                            mat = layer.Material
                            name = mat.Name if mat and hasattr(mat, 'Name') else "Unknown Layer Material"
                            materials.append(f"{name} (Layer)")
                elif hasattr(material_select, 'MaterialLayers'): # IfcMaterialLayerSet
                     for layer in material_select.MaterialLayers:
                         mat = layer.Material
                         name = mat.Name if mat and hasattr(mat, 'Name') else "Unknown LayerSet Material"
                         materials.append(f"{name} (LayerSet)")
                elif material_select.is_a("IfcMaterial"):
                    name = material_select.Name if hasattr(material_select, 'Name') else "Unknown Direct Material"
                    materials.append(f"{name} (Direct)")
                elif hasattr(material_select, 'Materials'): # IfcMaterialList
                     for mat in material_select.Materials:
                         name = mat.Name if mat and hasattr(mat, 'Name') else "Unknown List Material"
                         materials.append(f"{name} (List)")
                # Add checks for IfcMaterialConstituentSet etc. if needed

        # Fallback: Check for material property sets (less standard but common)
        if not materials:
             for definition in element.IsDefinedBy:
                  if definition.is_a('IfcRelDefinesByProperties'):
                      property_set = definition.RelatingPropertyDefinition
                      if property_set.is_a('IfcPropertySet') and hasattr(property_set, 'HasProperties'):
                          for prop in property_set.HasProperties:
                              if prop.is_a('IfcPropertySingleValue') and hasattr(prop, 'NominalValue') and prop.NominalValue:
                                  # Look for properties containing 'material' in their name
                                  prop_name = prop.Name if hasattr(prop, 'Name') else ""
                                  if "material" in prop_name.lower():
                                     val = prop.NominalValue.wrappedValue
                                     materials.append(f"{val} (Property: {prop_name})")

    except Exception as e:
        logging.warning(f"Error getting material for {element.GlobalId}: {e}", exc_info=False)
        pass # Ignore errors during material extraction

    return ", ".join(materials) if materials else "Undefined"

# Use Streamlit's caching for the expensive processing step
# Hash the file content to ensure re-run on file change
@st.cache_data(max_entries=5, show_spinner="Processing IFC model...")
def extract_ifc_data(_ifc_file_content_hash, ifc_file_path):
    """
    Processes the IFC file to extract geometry and metadata.
    _ifc_file_content_hash is used only for caching invalidation.
    """
    elements_for_threejs = []
    metadata_list = []
    ifc_file = None

    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
        elements = ifc_file.by_type("IfcProduct")
        total_elements = len(elements)
        logging.info(f"Starting processing of {total_elements} IfcProduct elements.")
        progress_bar = st.progress(0)
        processed_count = 0
        geocoded_count = 0

        for i, element in enumerate(elements):
            element_type = element.is_a()
            # Filter for target element types
            if element_type in TARGET_IFC_ELEMENTS:
                processed_count += 1
                try:
                    # Create shape geometry
                    shape = ifcopenshell.geom.create_shape(settings, element)
                    verts = shape.geometry.verts
                    faces = shape.geometry.faces

                    if not verts or not faces:
                        logging.debug(f"Skipping element {element.GlobalId} ({element_type}): No geometry generated.")
                        continue

                    # Convert geometry data for JSON serialization
                    vertices_list = [float(v) for v in verts]
                    faces_list = [int(f) for f in faces] # Indices should be integers

                    global_id = element.GlobalId if hasattr(element, 'GlobalId') else f"NO_GUID_{i}"
                    name = element.Name if hasattr(element, 'Name') and element.Name else element_type
                    material_desc = get_material_info(element) # Use cached function

                    # Data structure for Three.js
                    elements_for_threejs.append({
                        "id": global_id,
                        "type": element_type,
                        "name": name,
                        "material": material_desc,
                        "geometry": {
                            "vertices": vertices_list,
                            "faces": faces_list
                        }
                    })

                    # Data structure for Streamlit DataFrame
                    metadata_list.append({
                        "GlobalId": global_id,
                        "IFC Type": element_type,
                        "Name": name,
                        "Materials": material_desc,
                        "Vertices": len(vertices_list) // 3 if vertices_list else 0,
                        "Faces": len(faces_list) // 3 if faces_list else 0,
                    })
                    geocoded_count += 1

                except Exception as e:
                    logging.warning(f"Could not process element {element.GlobalId if hasattr(element, 'GlobalId') else 'N/A'} ({element_type}): {e}", exc_info=False)
            
            # Update progress bar
            progress = (i + 1) / total_elements
            try:
                progress_bar.progress(progress)
            except Exception:
                 pass # May fail if widget removed during processing


        progress_bar.empty() # Remove progress bar
        logging.info(f"Finished processing. Found {processed_count} target elements, successfully generated geometry for {geocoded_count}.")

        if not geocoded_count:
            st.warning("No geometric elements could be processed. The IFC file might be empty, contain unsupported geometry, or only contain non-target element types.")
            return [], pd.DataFrame()

        metadata_df = pd.DataFrame(metadata_list)
        return elements_for_threejs, metadata_df

    except Exception as e:
        logging.error(f"Fatal error processing IFC file: {e}", exc_info=True)
        st.error(f"An error occurred while processing the IFC file: {e}")
        return [], pd.DataFrame()
    finally:
        # Ensure the ifc_file object is handled correctly (though ifcopenshell might not need explicit closing)
        del ifc_file


# --- Three.js HTML/JS Component ---
def render_threejs_viewer(elements_json, component_key):
    """Renders the Three.js viewer using st.components.v1.html."""
    # Convert Python list of dicts to a JSON string for embedding in HTML
    elements_data_json_string = json.dumps(elements_json)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>BIM Viewer</title>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #ffffff; }} /* White background */
            canvas {{ display: block; }}
            #infoBox {{ /* Basic styling for a potential info box */
                position: absolute;
                top: 10px; left: 10px;
                padding: 5px 10px;
                background: rgba(0, 0, 0, 0.7); /* Dark semi-transparent */
                color: white;
                border-radius: 5px;
                font-family: sans-serif;
                font-size: 12px;
                display: none; /* Hidden by default */
            }}
        </style>
    </head>
    <body>
        <div id="viewer-container" style="width: 100%; height: 100%;"></div>
        <div id="infoBox"></div>

        <script type="importmap">
            {{
                "imports": {{
                    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
                }}
            }}
        </script>

        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            // --- Data from Streamlit ---
            const elementsData = JSON.parse({json.dumps(elements_data_json_string)}); // Parse the embedded JSON string
            const infoBox = document.getElementById('infoBox');
            let selectedObject = null; // Keep track of selected object

            // --- Basic Scene Setup ---
            const container = document.getElementById('viewer-container');
            if (!container) {{ console.error("Viewer container not found!"); return; }}

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff); // White background

            const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 10000); // Increased far plane
            camera.position.set(100, 100, 100); // Initial fallback position

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio); // Adjust for high-DPI screens
            container.appendChild(renderer.domElement);

            // --- Lighting (Enhanced for B&W) ---
            scene.add(new THREE.AmbientLight(0xeeeeee, 0.8)); // Brighter ambient light

            const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
            dirLight1.position.set(1, 1.5, 1).normalize();
            scene.add(dirLight1);

            const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight2.position.set(-1, -1, -0.5).normalize();
            scene.add(dirLight2);

            // --- Controls ---
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.1;
            controls.minDistance = 1; // Prevent zooming too close
            controls.maxDistance = 5000; // Allow zooming out far

            // --- Geometry Processing ---
            const meshes = [];
            const boundingBox = new THREE.Box3();
            const materialMap = new Map(); // Store materials by name to reuse them

            // Grayscale palette for materials
            const greyColors = [
                0x999999, 0xaaaaaa, 0xbbbbbb, 0xcccccc, 0xdddddd, 0xeeeeee,
                0x888888, 0x777777, 0x666666, 0x555555
             ];
            let colorIndex = 0;

            const defaultMaterial = new THREE.MeshStandardMaterial({{
                color: 0xcccccc, metalness: 0.1, roughness: 0.8, side: THREE.DoubleSide
            }});

            elementsData.forEach(element => {{
                if (!element || !element.geometry || !element.geometry.vertices || !element.geometry.faces || element.geometry.vertices.length === 0 || element.geometry.faces.length === 0) {{
                    console.warn(Skipping element ${'{element.id}'} - Missing or empty geometry data.);
                    return;
                }}

                try {{
                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(element.geometry.vertices, 3));
                    geometry.setIndex(element.geometry.faces);
                    geometry.computeVertexNormals(); // Crucial for lighting

                    let mat = defaultMaterial;
                    const materialName = element.material || "Undefined";

                    if (materialName !== "Undefined") {{
                        if (!materialMap.has(materialName)) {{
                            // Create a new grayscale material
                            const newColor = greyColors[colorIndex % greyColors.length];
                            colorIndex++;
                            mat = new THREE.MeshStandardMaterial({{
                                color: newColor,
                                metalness: 0.1,
                                roughness: 0.8,
                                side: THREE.DoubleSide,
                                name: materialName // Store name for reference
                            }});
                            materialMap.set(materialName, mat);
                        }} else {{
                            mat = materialMap.get(materialName);
                        }}
                    }}

                    const mesh = new THREE.Mesh(geometry, mat);
                    mesh.userData = {{
                        id: element.id,
                        type: element.type,
                        name: element.name,
                        material: materialName
                    }};
                    scene.add(mesh);
                    meshes.push(mesh);
                    boundingBox.expandByObject(mesh); // Calculate bounds based on mesh

                }} catch (error) {{
                    console.error(Error creating mesh for element ${'{element.id}'}:, error);
                }}
            }});

            // --- Center Camera ---
            if (!boundingBox.isEmpty()) {{
                const center = boundingBox.getCenter(new THREE.Vector3());
                const size = boundingBox.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraDist = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraDist *= 1.8; // Increase distance slightly for padding

                // Position camera further away and look at the center
                camera.position.copy(center);
                camera.position.addScaledVector(new THREE.Vector3(1, 0.8, 1).normalize(), cameraDist); // Offset position
                camera.near = Math.max(0.1, cameraDist / 100); // Adjust near plane
                camera.far = cameraDist * 5; // Adjust far plane
                camera.updateProjectionMatrix();

                controls.target.copy(center);
                console.log("Camera positioned based on bounding box.");
            }} else {{
                controls.target.set(0, 0, 0);
                 console.log("No bounding box data, using default camera position.");
            }}
            controls.update();


            // --- Basic Raycasting for Selection ---
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            function onClick( event ) {{
                // Adjust mouse coordinates relative to the container
                const rect = container.getBoundingClientRect();
                mouse.x = ( (event.clientX - rect.left) / container.clientWidth ) * 2 - 1;
                mouse.y = - ( (event.clientY - rect.top) / container.clientHeight ) * 2 + 1;

                raycaster.setFromCamera( mouse, camera );
                const intersects = raycaster.intersectObjects( meshes ); // Use the array of meshes

                if ( intersects.length > 0 ) {{
                    const firstIntersect = intersects[0].object;
                    if (selectedObject && selectedObject !== firstIntersect) {{
                        // Optional: Reset previously selected object's material if you implement highlighting
                    }}
                    selectedObject = firstIntersect;

                    // Log to console (primary feedback for now)
                    console.log("Selected:", selectedObject.userData);

                    // Update Info Box
                    infoBox.textContent = ID: ${'{selectedObject.userData.id}'}\\nType: ${'{selectedObject.userData.type}'}\\nName: ${'{selectedObject.userData.name}'}\\nMaterial: ${'{selectedObject.userData.material}'};
                    infoBox.style.display = 'block';

                    // --- IMPORTANT ---
                    // Sending data back to Streamlit from here requires a custom component.
                    // For now, info is only in the browser console and the temporary info box.
                    // -----------------

                }} else {{
                    // Clicked on empty space
                     if (selectedObject) {{
                         // Optional: Reset previously selected object's material
                     }}
                    selectedObject = null;
                    infoBox.style.display = 'none';
                    console.log("Clicked empty space.");
                }}
            }}
            container.addEventListener( 'click', onClick );


            // --- Animation Loop ---
            function animate() {{
                requestAnimationFrame(animate);
                controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true
                renderer.render(scene, camera);
            }}

            // --- Resize Handling ---
            function onWindowResize() {{
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}
            window.addEventListener('resize', onWindowResize);

            // Start rendering
            animate();
            console.log("Three.js viewer initialized.");

        </script>
    </body>
    </html>
    """
    return html_code

# --- Main Streamlit App Logic ---
def main():
    st.title("🏗️ BIM-CAI Dashboard")
    st.markdown("Upload an IFC model to view its geometry, metadata, and upcoming analyses.")

    uploaded_file = st.file_uploader("Choose an IFC file (.ifc)", type=['ifc'])

    if uploaded_file is not None:
        # Use a temporary file to work with ifcopenshell reliably
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Generate a hash of the file content for caching
        file_hash = hash(uploaded_file.getvalue())

        # Define tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 3D Viewer & Metadata",
            "💲 Cost Analysis",
            "⚠️ Clash Detection",
            "🌱 Environmental Prediction"
        ])

        with tab1:
            st.header("3D Model Viewer")
            st.caption("Interact with the model: Orbit (left-click & drag), Zoom (scroll), Pan (right-click & drag). Click elements for details (see browser console).")

            # Process the IFC data (this will be cached)
            elements_data, metadata_df = extract_ifc_data(file_hash, tmp_file_path)

            if elements_data:
                # Generate a unique key for the component based on file hash
                component_key = f"threejs_viewer_{file_hash}"
                
                # Embed the Three.js viewer HTML
                viewer_html = render_threejs_viewer(elements_data, component_key)
                st.components.v1.html(viewer_html, height=600, scrolling=False)

                st.divider()
                st.header("Extracted Element Metadata")
                with st.expander("Show/Hide Metadata Table", expanded=False):
                    if not metadata_df.empty:
                        # Improve display: Show fewer rows by default, allow download
                        st.dataframe(metadata_df)
                        st.download_button(
                            label="Download Metadata as CSV",
                            data=metadata_df.to_csv(index=False).encode('utf-8'),
                            file_name=f'{os.path.splitext(uploaded_file.name)[0]}_metadata.csv',
                            mime='text/csv',
                        )
                    else:
                        st.info("No metadata extracted for the targeted elements.")
            else:
                st.error("Could not display 3D model. Ensure the IFC file is valid and contains supported geometry for the targeted element types.")

        with tab2:
            st.header("Cost Analysis")
            st.info("Coming Soon: Estimated construction and operational costs.")
            # Placeholder for future cost analysis results

        with tab3:
            st.header("Clash Detection")
            st.info("Coming Soon: Identification of conflicts between building components.")
            # Placeholder for future clash detection results

        with tab4:
            st.header("Environmental Prediction (AI)")
            st.info("Coming Soon: AI-driven predictions for Energy Efficiency and Carbon Footprint.")
            # Placeholder for future environmental prediction results

        # Clean up the temporary file after processing
        try:
            os.remove(tmp_file_path)
        except Exception as e:
            logging.warning(f"Could not remove temporary file {tmp_file_path}: {e}")

    else:
        st.info("Please upload an IFC file to begin.")

if __name__ == "_main_":
    main()