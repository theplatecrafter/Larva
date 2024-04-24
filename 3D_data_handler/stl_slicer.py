import numpy as np
import ezdxf
from stl import mesh as stl_mesh
import os

def slice_stl_to_dwg(stl_filename, slicing_planes, output_prefix, output_dir='output'):
  os.makedirs(output_dir, exist_ok=True)

  stl_model = stl_mesh.Mesh.from_file(stl_filename)

  for plane_index, (axis, position) in enumerate(slicing_planes):
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    direction = np.sign(position)

    intersecting_polygons = []
    for triangle in stl_model.vectors:
      if (triangle[:, axis_index].min() * direction <= position * direction <= triangle[:, axis_index].max() * direction):
        intersecting_polygons.append(triangle)

    dxf = ezdxf.new("R2010")
    msp = dxf.modelspace()

    for polygon in intersecting_polygons:
      points = polygon[:, [0, 1]]
      points = np.vstack([points, points[0]])
      msp.add_lwpolyline(points)

    output_filename = f"{output_prefix}_{axis}{position}.dwg"
    output_path = os.path.join(output_dir, output_filename)
    dxf.saveas(output_path)
    print(f"Saved {len(intersecting_polygons)} polygons to {output_path}")

# Example usage
stl_filename = 'data/Another_Hollow_Cube.stl'
slicing_planes = [('x', 0), ('x', 1), ('x', 2)]
output_prefix = 'output_slice'
output_dir = 'output/CAD'
slice_stl_to_dwg(stl_filename, slicing_planes, output_prefix, output_dir)
