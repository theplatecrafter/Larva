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

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def LineSegmantPlaneCollision(planeNormal:np.ndarray, planePoint:np.ndarray,a0:np.ndarray,a1:np.ndarray):
    rayDirection = a1-a0
    rayPoint = a0
    planeNormal = np.array(planeNormal)
    planePoint = np.array(planePoint)

    if planeNormal.dot(planePoint-a0) == 0 and planeNormal.dot(planePoint-a1) == 0:
        return [a0,a1]

    point = LinePlaneCollision(planeNormal,planePoint,rayDirection,rayPoint)
    try:
        if list(point) != None:
            n = np.linalg.norm(point-a0)/np.linalg.norm(a1-a0)
            if 0 < n < 1:
                return point
    except:
        pass
    return np.array([None,None,None])

def SliceTriangleAtPlane(planeNormal:np.ndarray,planePoint:np.ndarray,triangle:list):
    a,b,c = triangle[0],triangle[1],triangle[2]
    check0, check1, check2 = LineSegmantPlaneCollision(planeNormal,planePoint,a,b), LineSegmantPlaneCollision(planeNormal,planePoint,c,b), LineSegmantPlaneCollision(planeNormal,planePoint,a,c)
    if len(check0) == 2 and len(check1) == 2 and len(check2) == 2:
        return [tuple(i) for i in triangle]
    elif len(check0) == 2:
        return [tuple(i) for i in check0]
    elif len(check1) == 2:
        return [tuple(i) for i in check1]
    elif len(check2) == 2:
        return [tuple(i) for i in check2]
    else:
        out = [tuple(i) for i in [check0,check1,check2] if (i != None).all()]
        if len(out) == 0:
            return None
        else:
            return out