import numpy as np

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def LineSegmantPlaneCollision(planeNormal:list, planePoint:list,line:list):
    a0 = np.array(line[0])
    a1 = np.array(line[1])
    rayDirection = a1-a0
    rayPoint = a0 #Any point along the ray
    planeNormal = np.array(planeNormal)
    planePoint = np.array(planePoint)

    if planeNormal.dot(planePoint-a0) == 0 and planeNormal.dot(planePoint-a1) == 0:
        return line

    point = LinePlaneCollision(planeNormal,planePoint,rayDirection,rayPoint)
    if point.any() != None:
        n = abs(point-a0)/abs(a1-a0)
        print(n)
        if 0 < n < 1:
            return tuple(point)
    
    return None

def TrianglePlaneCollision(triangle:list,planeNormal:list,planePoint:list):
    p0, p1, p2 = triangle[0], triangle[1], triangle[2]
    line0, line1, line2 = [p0,p1], [p0,p2], [p1,p2]
    out0, out1, out2 = LineSegmantPlaneCollision(planeNormal,planePoint,line0),LineSegmantPlaneCollision(planeNormal,planePoint,line1),LineSegmantPlaneCollision(planeNormal,planePoint,line2)
    if len(out0) == 2 and len(out1) == 2 and len(out2) == 2:
        return triangle
    elif len(out0) == 2:
        return out0
    elif len(out1) == 2:
        return out1
    elif len(out2) == 2:
        return out2
    else:
        out = [i for i in [out0,out1,out2] if i != None]
        if len(out) == 0:
            return None
        else:
            return out
        
print(TrianglePlaneCollision([(1,1,1),(4,1,4),(2.5,4,2.5)],(2,2,2),(1,1,-2)))