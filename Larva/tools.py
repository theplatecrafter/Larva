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
from itertools import combinations
import shutil
from moviepy.editor import ImageSequenceClip
import math
import xlsxwriter as xlsw
from matplotlib.colors import hsv_to_rgb
from scipy.spatial.transform import Rotation as R
from ezdxf.math import BoundingBox, Vec3

## local
def force_remove_all(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"Failed to remove {item_path}: {e}")

    print(f"All files and directories in {directory_path} have been removed.")

## from mislanious scripts
def rmSame(x: list) -> list:
    """removes any duplicated values"""
    y = []
    for i in x:
        if i not in y:
            y.append(i)
    return y

def png_to_mp4(image_folder:str, output_folder:str,filename:str = "movie", fps=None):
    filename += ".mp4"
    image_files = sorted([os.path.join(image_folder, img)
                          for img in os.listdir(image_folder)
                          if img.endswith(".png")])
    
    if fps == None:
        fps = int(math.log(len(image_files)+1,1.3))

    clip = ImageSequenceClip(image_files, fps=fps)
    
    clip.write_videofile(os.path.join(output_folder,filename), codec="libx264")

def createSimpleXLSX(collumNames:list,collumContent:list,output_folder:str,output_name:str="xls"):
    workbook = xlsw.Workbook(os.path.join(output_folder,output_name+".xlsx"))
    worksheet = workbook.add_worksheet()
    for i in range(len(collumNames)):
        worksheet.write(0,i,collumNames[i])
    for i in range(len(collumContent)):
        for j in range(len(collumContent[i])):
            worksheet.write(j+1,i,collumContent[i][j])
    
    workbook.close()

def printIF(boolean:bool,printString:str):
    """prints printString if boolean == True"""
    if boolean:
        print(printString)

## stl slicing
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(line1, line2):
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def is_valid_triangle(points:list,maxArea:float = 0):
    """
    Determine if a 2D triangle is valid based on its area.

    Args:
    points (list): A list of three tuples representing the triangle's vertices (x, y).

    Returns:
    bool: True if the triangle is valid (area > 0), False otherwise.
    """
    
    return calculateAreaOfTriangle(points) > maxArea

def calculateAreaOfTriangle(points:list):
    if len(points) != 3:
        raise ValueError("Input must be a list of three tuples representing the vertices of a triangle.")
    
    (x1, y1), (x2, y2), (x3, y3) = points[0],points[1],points[2]

    # Calculate the area using the determinant method
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return area

def create_dwg_outof_lines(lines,out):
    newDWG = ezdxf.new()
    msp = newDWG.modelspace()

    for line in lines:
        msp.add_line(line[0],line[1])
    
    newDWG.saveas(out)

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=0, printDeets=False):
    ndotu = np.dot(planeNormal, rayDirection)
    printIF(printDeets, f"ndotu: {ndotu}")
    if abs(ndotu) < epsilon:
        printIF(printDeets, "No intersection, the line is parallel to the plane.")
        return None

    w = rayPoint - planePoint
    si = -np.dot(planeNormal, w) / ndotu
    Psi = rayPoint + si * rayDirection
    printIF(printDeets, f"Collision Point: {Psi}")
    return Psi

def LineSegmentPlaneCollision(planeNormal: np.ndarray, planePoint: np.ndarray, a0: np.ndarray, a1: np.ndarray, printDeets=False):
    rayDirection = a1 - a0
    rayPoint = a0
    planeNormal = np.array(planeNormal)
    planePoint = np.array(planePoint)

    printIF(printDeets, f"Checking line segment: {a0} to {a1}")

    if np.isclose(np.dot(planeNormal, planePoint - a0), 0) and np.isclose(np.dot(planeNormal, planePoint - a1), 0):
        printIF(printDeets, "The line segment is on the plane.")
        return [tuple(a0), tuple(a1)]

    point = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, printDeets=printDeets)
    if point is not None:
        if is_point_on_segment(a0,a1,point):
            return tuple(point)

    return None

def is_point_on_segment(A, B, I, epsilon=0):
    # Check bounding box conditions
    in_box_x = min(A[0], B[0]) <= I[0] <= max(A[0], B[0])
    in_box_y = min(A[1], B[1]) <= I[1] <= max(A[1], B[1])
    in_box_z = min(A[2], B[2]) <= I[2] <= max(A[2], B[2])

    # Check if point I is within the bounding box
    if not (in_box_x and in_box_y and in_box_z):
        return False

    # Check for collinearity
    AB = B - A
    AI = I - A

    # Cross product to check if the direction is the same
    cross_product = np.cross(AB, AI)
    if not np.all(np.abs(cross_product) < epsilon):
        return False

    # Parameter t should be in [0, 1] for point I to be between A and B
    dot_product = np.dot(AI, AB)
    length_squared = np.dot(AB, AB)
    t = dot_product / length_squared

    return 0 <= t <= 1

def SliceTriangleAtPlane(planeNormal: np.ndarray, planePoint: np.ndarray, triangle: list, printDeets=False):
    a, b, c = np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2])
    printIF(printDeets, f"Triangle vertices: {a}, {b}, {c}")
    check0 = LineSegmentPlaneCollision(planeNormal, planePoint, a, b, printDeets=printDeets)
    check1 = LineSegmentPlaneCollision(planeNormal, planePoint, b, c, printDeets=printDeets)
    check2 = LineSegmentPlaneCollision(planeNormal, planePoint, c, a, printDeets=printDeets)

    printIF(printDeets, f"Intersections: {check0}, {check1}, {check2}")

    if check0 is None and check1 is None and check2 is None:
        return None

    points = []
    if check0 is not None and check1 is not None and check2 is not None:
        if np.isclose(np.dot(planeNormal, a - planePoint), 0) and np.isclose(np.dot(planeNormal, b - planePoint), 0) and np.isclose(np.dot(planeNormal, c - planePoint), 0):
            return [tuple(a), tuple(b), tuple(c)]
        points.extend([check0, check1, check2])
    else:
        if check0 is not None:
            points.append(check0)
        if check1 is not None:
            points.append(check1)
        if check2 is not None:
            points.append(check2)

    # Ensure all elements are tuples before converting to set
    flattened_points = []
    for point in points:
        if isinstance(point, list):
            flattened_points.extend(point)
        else:
            flattened_points.append(point)

    unique_points = list(set(flattened_points))
    
    if len(unique_points) == 2:
        return unique_points
    elif len(unique_points) > 2:
        return unique_points

    return None

def align_mesh_to_cutting_plane(stl_mesh, plane_normal, plane_point):
    # Normalize the plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Define the standard plane normal (z-axis)
    standard_normal = np.array([0, 0, 1])
    
    # Calculate the rotation axis (cross product of plane_normal and standard_normal)
    rotation_axis = np.cross(plane_normal, standard_normal)
    if np.linalg.norm(rotation_axis) == 0:
        # Vectors are parallel, no need to rotate
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Calculate the rotation angle (dot product and arccosine)
        dot_product = np.dot(plane_normal, standard_normal)
        rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        # Create the rotation matrix using the axis and angle
        rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()
    
    # Translate the mesh so that the plane point becomes the origin
    translation_vector = -plane_point
    
    # Apply the rotation and translation to the mesh
    stl_mesh.translate(translation_vector)
    stl_mesh.vectors = np.dot(stl_mesh.vectors, rotation_matrix.T)
    
    return stl_mesh

def signed_point_to_plane_distance(plane_normal, plane_point):
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    
    # Distance formula without absolute value to keep the sign
    distance = np.dot(plane_normal, plane_point) / np.linalg.norm(plane_normal)
    return distance

def find_signed_min_max_distances(points, plane_normal):
    min_distance = float('inf')
    max_distance = float('-inf')
    min_point = None
    max_point = None

    for point in points:
        distance = signed_point_to_plane_distance(plane_normal, point)
        
        if distance < min_distance:
            min_distance = distance
            min_point = point
        
        if distance > max_distance:
            max_distance = distance
            max_point = point
    
    return min_point, min_distance, max_point, max_distance

def get_entity_extents(entity):
    if entity.dxftype() == 'LWPOLYLINE':
        # Initialize extents with None to find minimum and maximum points
        min_point = None
        max_point = None
        
        for vertex in entity.vertices():
            # Vertex is a tuple (x, y[, z]) - handle both 2D and 3D vertices
            x, y, z = vertex[:3] if len(vertex) >= 3 else (vertex[0], vertex[1], 0)  # Assume z = 0 for 2D vertices
            
            if min_point is None:
                min_point = [x, y, z]
                max_point = [x, y, z]
            else:
                min_point[0] = min(min_point[0], x)
                min_point[1] = min(min_point[1], y)
                min_point[2] = min(min_point[2], z)
                max_point[0] = max(max_point[0], x)
                max_point[1] = max(max_point[1], y)
                max_point[2] = max(max_point[2], z)
        
        return min_point, max_point
    
    elif entity.dxftype() in ['LINE', 'POLYLINE']:
        # For LINE and POLYLINE, return the start and end points
        start_point = list(entity.dxf.start)[:3]
        end_point = list(entity.dxf.end)[:3]
        return start_point, end_point
    
    else:
        # Handle other entity types or return default extents
        return [0, 0, 0], [0, 0, 0]

def translate_entities(entities, translation):
    new_entities = []
    
    for entity in entities:
        if entity.dxftype() == 'LWPOLYLINE':
            new_entity = entity.copy()
            for i in range(len(new_entity)):
                new_entity[i] = (new_entity[i][0] + translation[0], new_entity[i][1] + translation[1], new_entity[i][2] + translation[2])
            new_entities.append(new_entity)
        else:
            start_point, end_point = get_entity_extents(entity)
            new_entity = entity.copy()
            new_entity.dxf.start = (start_point[0] + translation[0], start_point[1] + translation[1], start_point[2] + translation[2])
            new_entity.dxf.end = (end_point[0] + translation[0], end_point[1] + translation[1], end_point[2] + translation[2])
            new_entities.append(new_entity)
    
    return new_entities

## view
def view_stl_dev(stl_path: str, output_dir: str, output_name: str = "stl_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave:bool = False):
    target_stl = m.Mesh.from_file(stl_path)

    if start_end_points:
        xlim = [start_end_points[0][0],start_end_points[0][1]]
        ylim = [start_end_points[1][0],start_end_points[1][1]]
        zlim = [start_end_points[2][0],start_end_points[2][1]]
    else:
        min_x, max_x = np.min(target_stl.vectors[:, :, 0]), np.max(target_stl.vectors[:, :, 0])
        min_y, max_y = np.min(target_stl.vectors[:, :, 1]), np.max(target_stl.vectors[:, :, 1])
        min_z, max_z = np.min(target_stl.vectors[:, :, 2]), np.max(target_stl.vectors[:, :, 2])

        length_x = max_x - min_x
        length_y = max_y - min_y
        length_z = max_z - min_z

        largest_length = max(length_x, length_y, length_z)

        margin = largest_length * 0.1

        xlim = [(min_x + max_x - largest_length) / 2 - margin, (min_x + max_x + largest_length) / 2 + margin]
        ylim = [(min_y + max_y - largest_length) / 2 - margin, (min_y + max_y + largest_length) / 2 + margin]
        zlim = [(min_z + max_z - largest_length) / 2 - margin, (min_z + max_z + largest_length) / 2 + margin]

    x = 1600
    y = 1200
    fig = plt.figure(figsize=(x/100, y/100), dpi=100)

    ax = fig.add_subplot(111, projection='3d')

    polygons = []
    for i in range(len(target_stl.vectors)):
        tri = target_stl.vectors[i]
        polygons.append(tri)

    # Generate a colormap with unique colors
    cmap = plt.get_cmap('tab20', len(polygons))
    colors = [cmap(i) for i in range(len(polygons))]

    poly_collection = Poly3DCollection(polygons, linewidths=1, edgecolors='k')

    # Assign a different color to each triangle
    poly_collection.set_facecolors(colors)

    ax.add_collection3d(poly_collection)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(os.path.splitext(output_name)[0])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    if not onlySave:
        plt.show()
    plt.savefig(f"{output_dir}/{output_name}")

    if return_info:
        info = {
            'bounding_box': ((xlim[0], xlim[1]), (ylim[0], ylim[1]), (zlim[0], zlim[1])),
            'num_entities': len(target_stl.vectors),
            'filename': os.path.splitext(os.path.basename(stl_path))[0]
        }
        return plt
    return plt

def view_stl(stl_path: str, output_dir: str, output_name: str = "stl_view.png", 
             color_triangles: bool = True, keep_aspect_ratio: bool = True, 
             rotation_angles: tuple = None, resolution: tuple = (1600, 1200), margin: float = 0.1):
    # Load the STL file
    your_mesh = m.Mesh.from_file(stl_path)
    
    # Extract vertices and faces
    vertices = your_mesh.vectors.reshape(-1, 3)
    faces = your_mesh.vectors
    
    # Rotate the vertices if rotation_angles are provided
    if rotation_angles is not None:
        vertices = rotate(vertices, rotation_angles)
        faces = vertices.reshape(-1, 3, 3)
    
    # Create a new plot
    fig = plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Optionally color each triangle differently
    if color_triangles:
        face_colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
    else:
        face_colors = "blue"
    
    # Create a Poly3DCollection
    collection = Poly3DCollection(faces, facecolors=face_colors, edgecolor='k')
    ax.add_collection3d(collection)
    
    # Compute the range for each axis
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    
    # Compute the margins
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    z_margin = (z_max - z_min) * margin
    
    # Adjust aspect ratio if required
    if keep_aspect_ratio:
        # Find the largest dimension to scale equally
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Calculate midpoints
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2

        # Set the limits to keep aspect ratio with margins
        ax.set_xlim(mid_x - max_range / 2 - x_margin, mid_x + max_range / 2 + x_margin)
        ax.set_ylim(mid_y - max_range / 2 - y_margin, mid_y + max_range / 2 + y_margin)
        ax.set_zlim(mid_z - max_range / 2 - z_margin, mid_z + max_range / 2 + z_margin)
    else:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    # Automatically adjust camera angles if rotation_angles are not provided
    if rotation_angles is None:
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=20, azim=30)
    
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, output_name))
    plt.show()

def view_dwg(dwg_path: str, output_dir: str, output_name: str = "dwg_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave: bool = False, resolution: tuple = (1600, 1200)):
    dwg = ezdxf.readfile(dwg_path)
    msp = dwg.modelspace()
    entities = list(msp)
    
    start_x = end_x = start_y = end_y = None
    # Iterate over all entities to find the bounding box
    if start_end_points:
        start_x, start_y = start_end_points[0]
        end_x, end_y = start_end_points[1]
    else:
        for entity in entities:
            if entity.dxftype() == 'LINE':
                start_x = min(start_x, entity.dxf.start[0]) if start_x is not None else entity.dxf.start[0]
                end_x = max(end_x, entity.dxf.end[0]) if end_x is not None else entity.dxf.end[0]
                start_y = min(start_y, entity.dxf.start[1]) if start_y is not None else entity.dxf.start[1]
                end_y = max(end_y, entity.dxf.end[1]) if end_y is not None else entity.dxf.end[1]
            else:
                # Get all vertices of the entity
                vertices = entity.vertices()
                if vertices:
                    x_vals, y_vals = zip(*[(vertex[0], vertex[1]) for vertex in vertices])
                    start_x = min(start_x, min(x_vals)) if start_x is not None else min(x_vals)
                    end_x = max(end_x, max(x_vals)) if end_x is not None else max(x_vals)
                    start_y = min(start_y, min(y_vals)) if start_y is not None else min(y_vals)
                    end_y = max(end_y, max(y_vals)) if end_y is not None else max(y_vals)
        
        start_x -= (end_x - start_x) / 15
        end_x += (end_x - start_x) / 15
        start_y -= (end_y - start_y) / 15
        end_y += (end_y - start_y) / 15

    # Calculate figure size in inches based on the resolution and a DPI of 100
    fig_width = resolution[0] / 100
    fig_height = resolution[1] / 100

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    ax = fig.add_subplot(111)

    # Generate a colormap
    num_entities = len(entities)
    colors = hsv_to_rgb(np.linspace(0, 1, num_entities).reshape(-1, 1) * np.ones((1, 3)))

    for idx, entity in enumerate(entities):
        color = colors[idx]
        if entity.dxftype() == 'LINE':
            start_point = (entity.dxf.start[0], entity.dxf.start[1])
            end_point = (entity.dxf.end[0], entity.dxf.end[1])
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points()
            x = [point[0] for point in points]  # Extract x-coordinates
            y = [point[1] for point in points]  # Extract y-coordinates
            ax.plot(x, y, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    file_name, file_dot = os.path.splitext(output_name)
    ax.set_title(file_name)

    # Set plot display area based on start and end points
    if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
        ax.set_xlim(start_x, end_x)
        ax.set_ylim(start_y, end_y)

    if not onlySave:
        plt.show()
    plt.savefig(os.path.join(output_dir, output_name))
    
    if return_info:
        info = {
            'bounding_box': ((start_x, start_y), (end_x, end_y)),
            'num_entities': len(entities),
            'format': dwg.dxfversion,
            'filename': os.path.splitext(os.path.basename(dwg_path))[0]
        }
        return plt, info
    else:
        return plt

## other
def rotate(vertices, angles):
    # Convert angles from degrees to radians
    angles = np.radians(angles)
    
    # Rotation matrices around the x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    R_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    R_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # Rotate the vertices
    rotated_vertices = np.dot(vertices, R.T)
    return rotated_vertices
