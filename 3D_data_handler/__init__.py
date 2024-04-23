import numpy as np
from stl import mesh as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the STL file
test = m.Mesh.from_file('data/Another_Hollow_Cube.stl')

# Get the minimum and maximum coordinates for each axis
min_x, max_x = np.min(test.vectors[:,:,0]), np.max(test.vectors[:,:,0])
min_y, max_y = np.min(test.vectors[:,:,1]), np.max(test.vectors[:,:,1])
min_z, max_z = np.min(test.vectors[:,:,2]), np.max(test.vectors[:,:,2])

# Calculate the length of each axis
length_x = max_x - min_x
length_y = max_y - min_y
length_z = max_z - min_z

# Determine the largest length
largest_length = max(length_x, length_y, length_z)

# Calculate the margin based on the largest length
margin = largest_length * 0.1

# Set the axis limits
xlim = [(min_x + max_x - largest_length) / 2 - margin, (min_x + max_x + largest_length) / 2 + margin]
ylim = [(min_y + max_y - largest_length) / 2 - margin, (min_y + max_y + largest_length) / 2 + margin]
zlim = [(min_z + max_z - largest_length) / 2 - margin, (min_z + max_z + largest_length) / 2 + margin]

# Create a new figure
x = 1600
y = 1200
fig = plt.figure(figsize=(x/100, y/100), dpi=100)

# Create a new 3D axis
ax = fig.add_subplot(111, projection='3d')

# Create a list to store all the triangles
polygons = []

# Collect all the triangles into a single array
for i in range(len(test.vectors)):
    tri = test.vectors[i]
    polygons.append(tri)

# Add all the triangles to the plot in one go
ax.add_collection3d(Poly3DCollection(polygons, color='cyan', linewidths=1, edgecolors='blue'))


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mesh Visualization')

# Set plot limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

# Show the plot
plt.show()
