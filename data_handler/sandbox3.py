import main as dh
from tools import *

def  tmp(path_to_dwg:str):
    doc = ezdxf.readfile(path_to_dwg)
    msp = doc.modelspace()
    t = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    s = [entity for entity in msp if entity.dxftype() == "LINE"]

    for i in range(len(t)):
        p = t[i].get_points()
        if len(p) == 4:
            p.pop()
        for j in range(len(p)):
            p[j] = rmSame(p[j][:2])
    
    return p

print(tmp("output/CAD/dwg/cubeSliced1.dwg"))