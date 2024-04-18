import numpy as np
import ezdxf
from stl import mesh as stl_mesh  # Import the stl mesh module
import os

def slice_stl_to_dwg(stl_filename, slicing_planes, output_prefix, output_dir='output'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the STL file using the stl mesh module
    stl_model = stl_mesh.Mesh.from_file(stl_filename)

    # Iterate over slicing planes
    for plane_index, (axis, position) in enumerate(slicing_planes):
        # Determine slicing axis and direction
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        direction = np.sign(position)

        # Extract polygons intersecting with the slicing plane
        intersecting_polygons = []
        for triangle in stl_model.vectors:
            # Check if the triangle intersects with the slicing plane
            if (triangle[:, axis_index].min() * direction <= position * direction <= triangle[:, axis_index].max() * direction):
                intersecting_polygons.append(triangle)

        # Create a new DXF document
        dxf = ezdxf.new("R2007")
        msp = dxf.modelspace()

        # Add intersecting polygons to DXF
        for polygon in intersecting_polygons:
            points = polygon[:, [0, 1]]  # Extract X, Y coordinates
            # Append the first point to the end to close the polyline
            points = np.vstack([points, points[0]])
            msp.add_lwpolyline(points)

        # Save the DWG file to the output directory
        output_filename = f"{output_prefix}_{axis}{position}.dwg"
        output_path = os.path.join(output_dir, output_filename)
        dxf.saveas(output_path)
        print(f"Saved {len(intersecting_polygons)} polygons to {output_path}")

# Example usage
stl_filename = 'data/Stanford_Bunny_sample.stl'
slicing_planes = [('x', 0), ('y', 0), ('z', 0)]  # Example slicing planes (X=0, Y=0, Z=0)
output_prefix = 'output_slice'
output_dir = 'output'  # Output directory for DWG files
slice_stl_to_dwg(stl_filename, slicing_planes, output_prefix, output_dir)
