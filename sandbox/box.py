import Larva.data_handler as dh
from Larva.tools import *

def slice_stl_file_to_dwg(stl_mesh:m.Mesh,planeNormal:np.ndarray,planePoint:np.ndarray):
    rotateX(stl_mesh,np.array([0,0,0]),math.pi/2)


def rotateX(stl_mesh:m.Mesh,pivot:np.ndarray,angle:float):
    for i in range(len(stl_mesh.vectors)):
        print(stl_mesh.vectors[i])
        for j in range(len(stl_mesh.vectors[i])):
            stl_mesh.vectors[i][j] = rotate_point(stl_mesh.vectors[i][j],pivot,np.array([1,0,0]),angle)
        print(stl_mesh.vectors[i])

def rotateY(stl_mesh:m.Mesh,pivot:np.ndarray,angle:float):
    for i in range(len(stl_mesh.vectors)):
        print(stl_mesh.vectors[i])
        for j in range(len(stl_mesh.vectors[i])):
            stl_mesh.vectors[i][j] = rotate_point(stl_mesh.vectors[i][j],pivot,np.array([0,1,0]),angle)
        print(stl_mesh.vectors[i])
        
def rotateZ(stl_mesh:m.Mesh,pivot:np.ndarray,angle:float):
    for i in range(len(stl_mesh.vectors)):
        print(stl_mesh.vectors[i])
        for j in range(len(stl_mesh.vectors[i])):
            stl_mesh.vectors[i][j] = rotate_point(stl_mesh.vectors[i][j],pivot,np.array([0,0,1]),angle)
        print(stl_mesh.vectors[i])

def rotate_point(point:np.ndarray, pivot:np.ndarray, axis:np.ndarray, theta:float):
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