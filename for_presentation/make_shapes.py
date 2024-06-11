from Larva.tools import *

def view_stl(stl_path: str, output_dir: str, output_name: str = "stl_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave:bool = False):
    target_stl = m.Mesh.from_file(stl_path)

    if start_end_points:
        xlim = [start_end_points[0][0],start_end_points[0][1]]
        ylim = [start_end_points[1][0],start_end_points[1][1]]
        zlim = [start_end_points[2][0],start_end_points[2][1]]
    else:
        min_x, max_x = np.min(target_stl.vectors[:, :, 0]), np.max(target_stl.vectors[:, :, 0])
        min_y, max_y = np.min(target_stl.vectors[:, :, 1]), np.max(target_stl.vectors[:, :, 1])
        min_z, max_z = np.min(target_stl.vectors[:, :, 2]), np.max(target_stl.vectors[:, :, 2])

        length_x = max_x - min_x
        length_y = max_y - min_y
        length_z = max_z - min_z

        largest_length = max(length_x, length_y, length_z)

        margin = largest_length * 0.2

        xlim = [(min_x + max_x - largest_length) / 2 - margin, (min_x + max_x + largest_length) / 2 + margin]
        ylim = [(min_y + max_y - largest_length) / 2 - margin, (min_y + max_y + largest_length) / 2 + margin]
        zlim = [(min_z + max_z - largest_length) / 2 - margin, (min_z + max_z + largest_length) / 2 + margin]

    x = 1600
    y = 1200
    fig = plt.figure(figsize=(x/100, y/100), dpi=100)

    ax = fig.add_subplot(111, projection='3d')

    polygons = []
    for i in range(len(target_stl.vectors)):
        tri = target_stl.vectors[i]
        polygons.append(tri)

    # Generate a colormap with unique colors
    cmap = plt.get_cmap('tab20', len(polygons))
    colors = [cmap(i) for i in range(len(polygons))]

    poly_collection = Poly3DCollection(polygons, linewidths=1, edgecolors='k')

    # Assign a different color to each triangle
    poly_collection.set_facecolors(colors)

    ax.add_collection3d(poly_collection)

    # Add the plane after adding the object
    plane = np.array([
        [-57, 38, 10],
        [-57, 62, 10],
        [-33, 62, 10],
        [-33, 38, 10],
    ])
    ax.add_collection3d(Poly3DCollection([plane], facecolors='blue', edgecolors='k', alpha=0.5))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(os.path.splitext(output_name)[0])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    if not onlySave:
        plt.show()
    plt.savefig(f"{output_dir}/{output_name}")

    if return_info:
        info = {
            'bounding_box': ((xlim[0], xlim[1]), (ylim[0], ylim[1]), (zlim[0], zlim[1])),
            'num_entities': len(target_stl.vectors),
            'filename': os.path.splitext(os.path.basename(stl_path))[0]
        }
        return plt, info
    return plt



force_remove_all("for_presentation/out")
view_stl("data/stl/cube.stl","for_presentation/out")
