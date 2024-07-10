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

def rotateX(stl_mesh,pivot,angle):
    


path = "data/stl/cube.stl"
x = m.Mesh.from_file(path)
slice_stl_file_to_dwg(x,np.array([-1,0,1]),np.array([-55,40,0]))