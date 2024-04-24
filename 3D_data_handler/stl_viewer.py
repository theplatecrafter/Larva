import numpy as np
from stl import mesh as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

test = m.Mesh.from_file('data/Play_Brick_3mm.STL')

min_x, max_x = np.min(test.vectors[:,:,0]), np.max(test.vectors[:,:,0])
min_y, max_y = np.min(test.vectors[:,:,1]), np.max(test.vectors[:,:,1])
min_z, max_z = np.min(test.vectors[:,:,2]), np.max(test.vectors[:,:,2])

length_x = max_x - min_x
length_y = max_y - min_y
length_z = max_z - min_z

largest_length = max(length_x, length_y, length_z)

margin = largest_length * 0.1

xlim = [(min_x + max_x - largest_length) / 2 - margin, (min_x + max_x + largest_length) / 2 + margin]
ylim = [(min_y + max_y - largest_length) / 2 - margin, (min_y + max_y + largest_length) / 2 + margin]
zlim = [(min_z + max_z - largest_length) / 2 - margin, (min_z + max_z + largest_length) / 2 + margin]

x = 1600
y = 1200
fig = plt.figure(figsize=(x/100, y/100), dpi=100)

ax = fig.add_subplot(111, projection='3d')

polygons = []

for i in range(len(test.vectors)):
  tri = test.vectors[i]
  polygons.append(tri)

ax.add_collection3d(Poly3DCollection(polygons, color='cyan', linewidths=1, edgecolors='blue'))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mesh Visualization')

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

plt.show()
plt.savefig("output/image/mesh_view.png")
