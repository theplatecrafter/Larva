import numpy as np
from stl import mesh as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from skimage.measure import label
import ezdxf
import subprocess
from shapely.geometry import Polygon, MultiPolygon

# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"
obj_output_dir = "output/CAD/obj"
stl_output_dir = "output/CAD/stl"

# local functions


def remove_files_in_directory(directory_path):
    files = os.listdir(directory_path)
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def clear_output_paths():
    remove_files_in_directory(body_output_dir)
    remove_files_in_directory(dwg_output_dir)
    remove_files_in_directory(image_output_dir)
    remove_files_in_directory(obj_output_dir)
    remove_files_in_directory(stl_output_dir)


# global functions
def plt_image_saver(plt,output_dir:str,output_name:str = "image_folder"):
    os.makedirs(f"{output_dir}/{output_name}")

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
        subprocess.run(["meshconv", "-c", "step", stl_path,
                       "-o", step_path], check=True)
        print(f"Conversion successful. STEP file saved at '{step_path}'.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Conversion failed. {e}")
        return False

def view_dwg(dwg_path: str, output_dir: str, output_name: str = "dwg_view.png"):
    dwg = ezdxf.readfile(dwg_path)
    msp = dwg.modelspace()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for entity in msp:
        if entity.dxftype() == 'LINE':
            start_point = (entity.dxf.start[0], entity.dxf.start[1])
            end_point = (entity.dxf.end[0], entity.dxf.end[1])
            ax.plot([start_point[0], end_point[0]], [
                    start_point[1], end_point[1]], 'b-')
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points()
            x = [point[0] for point in points]  # Extract x-coordinates
            y = [point[1] for point in points]  # Extract y-coordinates
            ax.plot(x, y, 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('DWG Visualization')

    plt.show()
    plt.savefig(f"{output_dir}/{output_name}")

    return plt

def extract_bodies_from_stl(stl_path: str, output_dir: str, output_base_name: str = "Body_"):
    stl_mesh = m.Mesh.from_file(stl_path)
    vertices = stl_mesh.vectors.reshape((-1, 3))
    result = label(stl_mesh.vectors[:, :, 0])

    if isinstance(result, tuple):
        labeled_array, num_labels = result
    else:
        labeled_array = result
        num_labels = np.max(labeled_array)

    labeled_array = labeled_array.reshape((-1,))

    bodies = {}
    for label_idx in range(1, num_labels + 1):
        label_vertices = vertices[labeled_array == label_idx]
        body_mesh = m.Mesh(
            np.zeros(label_vertices.shape[0], dtype=m.Mesh.dtype))
        for i, vertex in enumerate(label_vertices):
            body_mesh.vectors[i] = vertex
        bodies[f'{output_base_name}{label_idx}'] = body_mesh

    print(f'Number of bodies extracted: {len(bodies)}')
    for body_name, body_mesh in bodies.items():
        output_filename = f'{output_dir}/{body_name}.stl'
        body_mesh.save(output_filename)
        print(f'Saved {body_name} to {output_filename}')

    return bodies

def slice_stl_to_dwg(stl_path: str, slicing_planes: list, output_dir: str, output_base_name: str = "sliced_stl_"):
    os.makedirs(output_dir, exist_ok=True)

    stl_model = m.Mesh.from_file(stl_path)

    n = 0
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
        n += 1
        output_filename = f"{output_base_name}{n}.dwg"
        output_path = os.path.join(output_dir, output_filename)
        dxf.saveas(output_path)
        print(f"Saved {len(intersecting_polygons)} polygons to {output_path}")

def view_stl(stl_path: str, output_dir: str, output_name: str = "stl_view.png"):
    target_stl = m.Mesh.from_file(stl_path)

    min_x, max_x = np.min(target_stl.vectors[:, :, 0]), np.max(
        target_stl.vectors[:, :, 0])
    min_y, max_y = np.min(target_stl.vectors[:, :, 1]), np.max(
        target_stl.vectors[:, :, 1])
    min_z, max_z = np.min(target_stl.vectors[:, :, 2]), np.max(
        target_stl.vectors[:, :, 2])

    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z

    largest_length = max(length_x, length_y, length_z)

    margin = largest_length * 0.1

    xlim = [(min_x + max_x - largest_length) / 2 - margin,
            (min_x + max_x + largest_length) / 2 + margin]
    ylim = [(min_y + max_y - largest_length) / 2 - margin,
            (min_y + max_y + largest_length) / 2 + margin]
    zlim = [(min_z + max_z - largest_length) / 2 - margin,
            (min_z + max_z + largest_length) / 2 + margin]

    x = 1600
    y = 1200
    fig = plt.figure(figsize=(x/100, y/100), dpi=100)

    ax = fig.add_subplot(111, projection='3d')

    polygons = []

    for i in range(len(target_stl.vectors)):
        tri = target_stl.vectors[i]
        polygons.append(tri)

    ax.add_collection3d(Poly3DCollection(
        polygons, color='cyan', linewidths=1, edgecolors='blue'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Visualization')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.show()
    plt.savefig(f"{output_dir}/{output_name}")

    return plt

def obj_to_stl(obj_path: str, output_dir: str, output_name: str = "obj_converted.stl"):
    vertices = []
    faces = []

    # Load vertices and faces from the OBJ file
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                if len(face) != 3:  # Check if face has exactly 3 vertices
                    print(f"Skipping face with invalid number of vertices: {face}")
                else:
                    faces.append(face)

    # Check if any faces were loaded
    if not faces:
        print("No valid faces found in the OBJ file.")
        return

    # Convert vertices and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create an STL mesh
    mesh_data = m.Mesh(np.zeros(len(faces), dtype=m.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[f[j], :]

    # Save the mesh to an STL file
    output_file = os.path.join(output_dir, output_name)
    mesh_data.save(output_file)

def view_obj(obj_path: str, output_dir: str, output_base_name: str = "obj_view.png"):
    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)

    if len(vertices) == 0:
        print("No vertices found in the .obj file.")
        return

    vertices = np.array(vertices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(f"{output_dir}/{output_base_name}")
    plt.show()

    return plt

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