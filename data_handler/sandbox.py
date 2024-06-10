from tools import *


tri = [np.array([1,0,-0.00000032]),np.array([0,1,0.00000032]),np.array([0,0,0.00000032])]
out = SliceTriangleAtPlane(np.array([0,0,1]),np.array([0,0,0]),tri,True)
print(out)

if out != None:
    for i in range(len(out)):
        out[i] = out[i][:-1]
    print(out)