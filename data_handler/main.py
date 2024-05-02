import numpy as np
from stl import mesh as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from skimage.measure import label
import ezdxf
from step_converter import convert_stl_to_step

# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"

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


# global functions
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


def find_outer_edges(vertices):
    edges = set()
    for i in range(len(vertices)):
        edge = (tuple(vertices[i]), tuple(vertices[(i + 1) % len(vertices)]))
        reverse_edge = edge[::-1]
        if reverse_edge in edges:
            edges.remove(reverse_edge)
        else:
            edges.add(edge)
    return list(edges)

def slice_stl_to_dwg_without_lines(stl_path: str, slicing_planes: list, output_dir: str, output_base_name: str = "sliced_stl_"):
    os.makedirs(output_dir, exist_ok=True)

    stl_model = m.Mesh.from_file(stl_path)

    # Convert to STEP format
    step_filename = os.path.join(output_dir, "converted_model.step")
    convert_stl_to_step(stl_path, step_filename)

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
            edges = find_outer_edges(polygon)
            for edge in edges:
                msp.add_line(edge[0], edge[1])

        n += 1
        output_filename = f"{output_base_name}{n}.dwg"
        output_path = os.path.join(output_dir, output_filename)
        dxf.saveas(output_path)
        print(f"Saved {len(intersecting_polygons)} polygons to {output_path}")






