import Larva.direct_data_handler as dh
import trimesh
import numpy as np
import ezdxf
import os

import trimesh
import ezdxf
import numpy as np
import os

import trimesh
import ezdxf
import numpy as np
import os

import trimesh
import ezdxf
import numpy as np
import os

import trimesh
import ezdxf
import numpy as np
import os

def slice_3d_stl(input_file, output_file, slice_height):
    """
    Slice a 3D STL file at a specified height and export the result to a DWG file.
    
    :param input_file: Path to the input 3D STL file
    :param output_file: Path to save the output DWG file
    :param slice_height: Height at which to slice the object
    """
    # Load the STL file
    mesh = trimesh.load_mesh(input_file)
    print(f"Loaded mesh from {input_file}")

    # Create a plane for slicing
    slice_origin = [0, 0, slice_height]
    slice_normal = [0, 0, 1]

    # Slice the mesh
    slice_3d = mesh.section(plane_origin=slice_origin, plane_normal=slice_normal)

    if slice_3d is None or len(slice_3d.entities) == 0:
        print(f"No intersection at height {slice_height}")
        return

    print(f"Slice at height {slice_height} has {len(slice_3d.entities)} entities")

    # Create a new DWG file
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Add the slice to the DWG file
    for entity in slice_3d.entities:
        if isinstance(entity, trimesh.path.entities.Line):
            start, end = entity.end_points
            print(f"Line start: {start}, end: {end}")

            # Convert start and end points to tuples of floats if they are numpy types
            if isinstance(start, np.ndarray):
                start = start.tolist()
            if isinstance(end, np.ndarray):
                end = end.tolist()

            # Handle case where start and end might be scalar or list
            if isinstance(start, (list, np.ndarray)):
                start = tuple(float(coord) for coord in start[:2])
            elif isinstance(start, (float, int)):
                start = (float(start), float(start))  # Convert scalar to tuple

            if isinstance(end, (list, np.ndarray)):
                end = tuple(float(coord) for coord in end[:2])
            elif isinstance(end, (float, int)):
                end = (float(end), float(end))

            # Check for correct format
            if len(start) != 2 or len(end) != 2:
                raise ValueError(f"Unexpected format for start or end points: start={start}, end={end}")

            msp.add_line(start, end)
        elif isinstance(entity, trimesh.path.entities.Arc):
            center = entity.center
            radius = entity.radius
            start_angle = entity.angles[0]
            end_angle = entity.angles[1]
            print(f"Arc center: {center}, radius: {radius}, angles: {start_angle} to {end_angle}")

            # Convert center to tuple of floats
            if isinstance(center, np.ndarray):
                center = center.tolist()
            center = tuple(float(coord) for coord in center[:2])

            msp.add_arc(center, radius, start_angle, end_angle)
        else:
            print(f"Unsupported entity type: {type(entity)}")

    # Save the DWG file
    doc.saveas(output_file)
    print(f"Slice at height {slice_height} saved to {output_file}")
















def create_multiple_slices(input_file, output_folder, slice_width):
    """
    Create multiple slices of a 3D STL file and save them as separate DWG files.
    
    :param input_file: Path to the input 3D STL file
    :param output_folder: Folder to save the output DWG files
    :param slice_width: Width of each slice
    """
    # Load the STL file
    mesh = trimesh.load_mesh(input_file)

    # Get the bounding box of the mesh
    bounds = mesh.bounds
    zmin, zmax = bounds[:, 2]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate number of slices
    num_slices = int((zmax - zmin) / slice_width) + 1

    # Create slices
    for i in range(num_slices):
        slice_height = zmin + i * slice_width
        output_file = os.path.join(output_folder, f"slice_{i:03d}.dwg")
        slice_3d_stl(input_file, output_file, slice_height)

    print(f"Created {num_slices} slices in {output_folder}")

def combine_dwg_files(input_folder, output_file):
    """
    Combine multiple DWG files into a single DWG file, minimizing the overall size.
    
    :param input_folder: Path to the folder containing input DWG files
    :param output_file: Path to save the combined output DWG file
    """
    # Create a new DWG file
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Get all DWG files in the input folder
    dwg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dwg')]

    # Variables to track the overall bounding box
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Read and combine all DWG files
    for filename in dwg_files:
        filepath = os.path.join(input_folder, filename)
        src_doc = ezdxf.readfile(filepath)
        src_msp = src_doc.modelspace()

        for entity in src_msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                min_x = min(min_x, start[0], end[0])
                min_y = min(min_y, start[1], end[1])
                max_x = max(max_x, start[0], end[0])
                max_y = max(max_y, start[1], end[1])
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                min_x = min(min_x, center[0] - radius)
                min_y = min(min_y, center[1] - radius)
                max_x = max(max_x, center[0] + radius)
                max_y = max(max_y, center[1] + radius)

    # Calculate offset to move entities to (0,0)
    offset_x, offset_y = -min_x, -min_y

    # Add entities to the new document with offset
    for filename in dwg_files:
        filepath = os.path.join(input_folder, filename)
        src_doc = ezdxf.readfile(filepath)
        src_msp = src_doc.modelspace()

        for entity in src_msp:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                msp.add_line(
                    (start[0] + offset_x, start[1] + offset_y),
                    (end[0] + offset_x, end[1] + offset_y)
                )
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                msp.add_arc(
                    (center[0] + offset_x, center[1] + offset_y),
                    entity.dxf.radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle
                )

    # Save the combined DWG file
    doc.saveas(output_file)
    print(f"Combined DWG file saved as {output_file}")
    print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"Total width: {max_x - min_x}, Total height: {max_y - min_y}")




dh.clear_output_paths()
create_multiple_slices("data/stl/cube.stl", "output/slices", 1.0)
combine_dwg_files("output/slices", "output/CAD/dwg/combined_slices.dwg")
dh.view_dwg("output/slices/slice_001.dwg",dh.image_output_dir)