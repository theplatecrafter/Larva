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

def rmSame(x: list) -> list:
    """removes any duplicated values"""
    y = []
    for i in x:
        if i not in y:
            y.append(i)
    return y

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(line1, line2):
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def create_dwg_outof_lines(lines,out):
    newDWG = ezdxf.new()
    msp = newDWG.modelspace()

    for line in lines:
        msp.add_line(line[0],line[1])
    
    newDWG.saveas(out)

def png_to_mp4(image_folder:str, output_folder:str,filename:str = "movie", fps=None):
    filename += ".mp4"
    image_files = sorted([os.path.join(image_folder, img)
                          for img in os.listdir(image_folder)
                          if img.endswith(".png")])
    
    if fps == None:
        fps = int(math.log(len(image_files)+1,1.3))

    clip = ImageSequenceClip(image_files, fps=fps)
    
    clip.write_videofile(os.path.join(output_folder,filename), codec="libx264")

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
        n = np.linalg.norm(point - a0) / np.linalg.norm(a1 - a0)
        printIF(printDeets, f"Intersection parameter n: {n}")
        if 0 <= n <= 1:
            return tuple(point)

    return None

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
