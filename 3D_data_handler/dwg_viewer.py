import ezdxf
import matplotlib.pyplot as plt

dwg = ezdxf.readfile("output/CAD/output_slice_x2.dwg")
msp = dwg.modelspace()

fig = plt.figure()
ax = fig.add_subplot(111)

for entity in msp:
  if entity.dxftype() == 'LINE':
    start_point = entity.dxf.start[:2]
    end_point = entity.dxf.end[:2]
    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-')
  elif entity.dxftype() == 'LWPOLYLINE':
    points = entity.get_points()
    x = [point[0] for point in points[:2]]  # Extract first two x-coordinates
    y = [point[1] for point in points[:2]]  # Extract first two y-coordinates
    ax.plot(x, y, 'r-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('DWG Visualization')

plt.show()
plt.savefig("output/image/dwg_view.png")