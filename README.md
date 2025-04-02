# BIM-CAI Project

This project is a Building Information Modeling (BIM) tool that integrates Cost Analysis and Intelligent Clash Detection.

## Current Functionality

*   **Basic Dashboard:** Provides a user interface for interacting with the BIM model.
*   **3D Model Viewing:**  Allows users to view the 3D model of the building.
*   **Data and Cost Visualization:** Displays relevant data and cost information associated with the BIM model directly in the dashboard.
*   **Clash Detection:**  Highlights potential clashes or conflicts within the model.

## How to Run the Code
#Make sure to use Python3.12.0 for the code and activation of environment in venv as ifcopenshell isn't compatbile with the latest

These instructions assume you have Python installed. We recommend using a virtual environment.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Fusion831/BIM-CAI.git
    cd BIM-CAI
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**

    ```bash
    python main.py  # Or whatever the main script is called
    ```


5. **Access the Dashboard:**

   Open your web browser and navigate to the address shown in the console upon starting the application.

## TODO List

*   [ ] Find the data required to train the ML Models
*   [ ] EDA of the supposed model and then get the graphs (Data is hard to get)
