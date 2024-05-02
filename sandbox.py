import numpy as np
from stl import mesh
import ezdxf
import os
from shapely.geometry import Polygon, MultiPolygon

def slice_obj_file(input_file, slice_thickness, output_dir):
    # Load the OBJ file
    vertices = []
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('l '):
                edge = [int(v) for v in line.strip().split()[1:]]
                edges.append(edge)

    vertices = np.array(vertices)
    edges = np.array(edges) - 1  # Obj files use 1-based indexing

    # Determine the bounding box of the object
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    bounding_box = max_bound - min_bound
    num_slices = int(bounding_box[2] / slice_thickness)

    # Create slices
    for i in range(num_slices):
        z_min = min_bound[2] + i * slice_thickness
        z_max = z_min + slice_thickness

        # Find edges intersecting the slice
        intersecting_edges = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            if v1[2] <= z_max and v2[2] >= z_min or v2[2] <= z_max and v1[2] >= z_min:
                intersecting_edges.append([v1, v2])

        # Create polygons for the slice
        polygons = []
        for edge in intersecting_edges:
            x1, y1, z1 = edge[0]
            x2, y2, z2 = edge[1]
            # Project the edge onto the XY plane
            if z1 < z_min:
                x1 = x2 - (x2 - x1) * (z2 - z_min) / (z2 - z1)
                y1 = y2 - (y2 - y1) * (z2 - z_min) / (z2 - z1)
                z1 = z_min
            if z2 > z_max:
                x2 = x1 + (x2 - x1) * (z_max - z1) / (z2 - z1)
                y2 = y1 + (y2 - y1) * (z_max - z1) / (z2 - z1)
                z2 = z_max
            polygons.append(Polygon([(x1, y1), (x2, y2)]))

        # Create a DXF file for this slice
        dxf = ezdxf.new()
        msp = dxf.modelspace()

        for polygon in polygons:
            msp.add_lwpolyline(list(polygon.exterior.coords), close=True)

        output_file = os.path.join(output_dir, f"slice_{i}.dwg")
        dxf.saveas(output_file)

if __name__ == "__main__":
    input_file = "data/obj/teddy_bear.obj"  # Change to your input OBJ file path
    slice_thickness = 5  # Adjust as needed
    output_dir = "output/CAD/dwg"  # Specify the output directory here
    slice_obj_file(input_file, slice_thickness, output_dir)
