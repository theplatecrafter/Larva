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