import Larva.data_handler as dh
from Larva.tools import *

def slice_stl_file_to_dwg(stl_mesh:m.Mesh,planeNormal:np.ndarray,planePoint:np.ndarray):
    triDATA = []
    outTRI = []
    for triangle in stl_mesh.vectors:
        triDATA.append(SliceTriangleAtPlane(planeNormal,planePoint,triangle))
    triDATA = [i for i in triDATA if len(i) == 2]
    baseSegment = triDATA[0]
    baseLength = np.linalg.norm(baseSegment[1]-baseSegment[0])
    outTRI.append([np.array([0,0,0]),np.array([baseLength,0,0])])

def rotateX(stl_mesh:m.Mesh,pivot:np.ndarray,angle:float):
    new_triangles = []
    for triangle in stl_mesh.vectors:
        new_triangle = []
        new_triangle.append(triangle[0]*math.cos(angle)+(np.cross(pivot,triangle[0])*math.sin(angle)+pivot))

def rotate_point(point, pivot, axis, theta):
    """
    Rotate a point around a pivot by theta degrees along the given axis.

    Parameters:
    - point: (x, y, z) coordinates of the point to rotate.
    - pivot: (a, b, c) coordinates of the pivot point.
    - axis: (u_x, u_y, u_z) unit vector of the axis of rotation.
    - theta: angle in degrees to rotate.

    Returns:
    - (x_new, y_new, z_new): coordinates of the rotated point.
    """
    # Convert degrees to radians
    theta = np.radians(theta)

    # Convert to numpy arrays
    point = np.array(point)
    pivot = np.array(pivot)
    axis = np.array(axis)

    # Translate point to origin
    point -= pivot

    # Rodrigues' rotation formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    k = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector

    rotated_point = (point * cos_theta +
                     np.cross(k, point) * sin_theta +
                     k * np.dot(k, point) * (1 - cos_theta))

    # Translate point back
    rotated_point += pivot

    return rotated_point

path = "data/stl/cube.stl"
x = m.Mesh.from_file(path)
slice_stl_file_to_dwg(x,np.array([-1,0,1]),np.array([-55,40,0]))