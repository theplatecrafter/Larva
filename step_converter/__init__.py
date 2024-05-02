import os
import subprocess

def convert_stl_to_step(stl_path, step_path):
    """
    Convert an STL file to a STEP file.

    Args:
        stl_path (str): Path to the input STL file.
        step_path (str): Path to save the output STEP file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    if not os.path.isfile(stl_path):
        print(f"Error: File '{stl_path}' not found.")
        return False

    try:
        subprocess.run(["meshconv", "-c", "step", stl_path, "-o", step_path], check=True)
        print(f"Conversion successful. STEP file saved at '{step_path}'.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Conversion failed. {e}")
        return False
