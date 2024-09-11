import numpy as np
from stl import mesh as m
import math
import os
import trimesh
import networkx as nx
from ezdxf.math import Vec2
import ezdxf
import shutil
from scipy.spatial.transform import Rotation as R


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import label
from moviepy.editor import ImageSequenceClip
import xlsxwriter as xlsw
from matplotlib.colors import hsv_to_rgb


from .read_terminal import *

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

def calculateAreaOfTriangle(points:list,printDeets:bool = False):
    if len(points) != 3:
        raise ValueError("Input must be a list of three tuples representing the vertices of a triangle.")
    
    (x1, y1), (x2, y2), (x3, y3) = points[0],points[1],points[2]

    # Calculate the area using the determinant method
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    printIF(printDeets,f"triangle of points {points} has area of {area}","calculateAreaOfTriangle")
    return area

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=0, printDeets=False):
    ndotu = np.dot(planeNormal, rayDirection)
    printIF(printDeets, f"ndotu: {ndotu}")
    if abs(ndotu) < epsilon:
        printIF(printDeets, "No intersection, the line is parallel to the plane.","LinePlaneCollision")
        return None

    w = rayPoint - planePoint
    si = -np.dot(planeNormal, w) / ndotu
    Psi = rayPoint + si * rayDirection
    printIF(printDeets, f"Collision Point {Psi}","LinePlaneCollision")
    return Psi

def LineSegmentPlaneCollision(planeNormal: np.ndarray, planePoint: np.ndarray, a0: np.ndarray, a1: np.ndarray, printDeets=False):
    rayDirection = a1 - a0
    rayPoint = a0
    planeNormal = np.array(planeNormal)
    planePoint = np.array(planePoint)

    printIF(printDeets, f"Checking line segment: {a0} to {a1}","LineSegmentPlaneCollision")

    if np.isclose(np.dot(planeNormal, planePoint - a0), 0) and np.isclose(np.dot(planeNormal, planePoint - a1), 0):
        printIF(printDeets, "The line segment is on the plane.","LineSegmentPlaneCollision")
        return [tuple(a0), tuple(a1)]

    point = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, printDeets)
    printIF(printDeets, f"intersection point {point}","LineSegmentPlaneCollision")
    if point is not None:
        if is_point_on_segment(a0,a1,point):
            printIF(printDeets, f"{point} is between {a0} and {a1}","LineSegmentPlaneCollision")
            return tuple(point)
    
    printIF(printDeets, f"{point} is not between {a0} and {a1}","LineSegmentPlaneCollision")

    return None

def is_point_on_segment(A, B, I, epsilon=1e-8):
    # Check bounding box conditions
    in_box_x = min(A[0], B[0]) - epsilon <= I[0] <= max(A[0], B[0]) + epsilon
    in_box_y = min(A[1], B[1]) - epsilon <= I[1] <= max(A[1], B[1]) + epsilon
    in_box_z = min(A[2], B[2]) - epsilon <= I[2] <= max(A[2], B[2]) + epsilon

    # Check if point I is within the bounding box
    if not (in_box_x and in_box_y and in_box_z):
        return False

    # Check for collinearity using dot products
    AB = B - A
    AI = I - A
    BI = I - B

    # If the point I is collinear with A and B, then the vectors AI and BI should be oppositely directed
    collinear = np.dot(AI, AB) >= 0 and np.dot(BI, AB) <= 0
    if not collinear:
        return False

    # If all conditions are met, then I is on the line segment AB
    return True

def SliceTriangleAtPlane(planeNormal: np.ndarray, planePoint: np.ndarray, triangle: list, printDeets=False):
    a, b, c = np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2])
    printIF(printDeets, f"Triangle vertices: {a}, {b}, {c}    slicing plane normal = {planeNormal} slicing plane point = {planePoint}","SliceTriangleAtPlane")
    check0 = LineSegmentPlaneCollision(planeNormal, planePoint, a, b, printDeets=printDeets)
    check1 = LineSegmentPlaneCollision(planeNormal, planePoint, b, c, printDeets=printDeets)
    check2 = LineSegmentPlaneCollision(planeNormal, planePoint, c, a, printDeets=printDeets)

    printIF(printDeets, f"Intersections: {check0}, {check1}, {check2}","SliceTriangleAtPlane")

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

def align_mesh_to_cutting_plane(stl_mesh, plane_normal, plane_point,printDeets=False):
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
    printIF(printDeets,f"translation vector: {translation_vector}   rotation axis: {rotation_axis}","align_mesh_to_cutting_plane")
    
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


## plane combine
def get_entity_extents(entity,printDeets=False):
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
        printIF(printDeets,f"entity is LWPOLYLINE. min points: {min_point}, max points: {max_point}","get_entity_extents")
        
        return min_point, max_point
    elif entity.dxftype() in ['LINE', 'POLYLINE']:
        # For LINE and POLYLINE, return the start and end points
        start_point = list(entity.dxf.start)[:3]
        end_point = list(entity.dxf.end)[:3]
        printIF(printDeets,f"entity is LINE or POLYLINE. start point: {start_point}, end point: {end_point}","get_entity_extents")
        return start_point, end_point
    else:
        printIF(printDeets,"could not read entity","get_entity_extents")
        # Handle other entity types or return default extents
        return [0, 0, 0], [0, 0, 0]

def translate_entities(entities, translation,printDeets=False):
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
    
    printIF(printDeets,f"translated entities to {translation}","translate_entities")
    return new_entities

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

def print_styled(text, color_code, style_code="0",end="\n"):
    print(f"\033[{style_code};{color_code}m{text}\033[0m",end=end)

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

def printIF(boolean:bool,printString:str,precursor:str = "sys"):
    """prints printString if boolean == True"""
    if boolean:
        print_styled(precursor+":",33,"1",end=" ")
        print(printString)
