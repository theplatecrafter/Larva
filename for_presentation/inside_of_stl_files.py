from Larva.tools import *    ## 道具はこの行で省略

## run with: python -m for_presentation.inside_of_stl_files

def get_stl_info(path_to_stl:str,output_path:str,outputfoldername:str = "stl_file_infos"):
    output_path = os.path.join(output_path,outputfoldername)
    os.mkdir(output_path)
    body_path = os.path.join(output_path,"bodies")
    body_image_path = os.path.join(output_path,"body_images")
    os.mkdir(body_path)
    os.mkdir(body_image_path)
    stl_model = m.Mesh.from_file(path_to_stl)
    out = view_stl(path_to_stl,output_path,"stl.png",None,True,True)
    bounding_box = out[1]["bounding_box"]

    triangles_path = os.path.join(output_path,"triangles")
    triangles_image_path = os.path.join(output_path,"triangle_images")
    os.mkdir(triangles_image_path)

    return_stl_parts(path_to_stl,output_path,"triangles")
    files = [i for i in os.listdir(triangles_path) if i.lower().endswith(".stl")]
    triangles= [[i+1 for i in range(len(files))],[],[],[]]
    for i in range(len(files)):
        file_path = os.path.join(triangles_path,files[i])
        target_stl = m.Mesh.from_file(file_path)
        triangles[1].append(str(target_stl.vectors[0][0]))
        triangles[2].append(str(target_stl.vectors[0][1]))
        triangles[3].append(str(target_stl.vectors[0][2]))
        view_stl(file_path,triangles_image_path,f"triangle{i+1}.png",bounding_box,False,True)
        print(f"{i+1}/{len(files)} triangles loaded")

    png_to_mp4(triangles_image_path,output_path,"triangles")

    createSimpleXLSX(["#","p1","p2","p3"],triangles,output_path,"info")

    extract_bodies_from_stl(path_to_stl,body_path)
    files = [i for i in os.listdir(body_path) if i.lower().endswith(".stl")]
    for i in range(len(files)):
        file_path = os.path.join(body_path,files[i])
        view_stl(file_path,body_image_path,f"Body_{i}.png",bounding_box,False,True)
        print(f"Created image for body {i+1}/{len(files)}")
    png_to_mp4(body_image_path,output_path,"bodies")

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

        margin = largest_length * 0.1

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

def return_stl_parts(path_to_stl: str, outputdir: str, foldername: str = "stlparts"):
    try:
        original_mesh = m.Mesh.from_file(path_to_stl)
    except IOError:
        print(f"Could not read file: {path_to_stl}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    folder = os.path.join(outputdir, foldername)
    os.makedirs(folder, exist_ok=True)

    num_triangles = len(original_mesh)
    
    for i, triangle in enumerate(original_mesh.vectors):
        print(f"{i+1}/{num_triangles} triangles processed")

        # Create a new mesh with a single triangle
        new_mesh = m.Mesh(np.zeros(1, dtype=m.Mesh.dtype))
        new_mesh.vectors[0] = triangle.reshape((3, 3))

        # Save the new mesh as an STL file
        new_mesh.save(os.path.join(folder, f"triangle_{i+1}.stl"))

def extract_bodies_from_stl(stl_path: str, output_dir: str, output_base_name: str = "Body_"):
    stl_mesh = m.Mesh.from_file(stl_path)
    vertices = stl_mesh.vectors.reshape((-1, 3))
    result = label(stl_mesh.vectors[:, :, 0])

    if isinstance(result, tuple):
        labeled_array, num_labels = result
    else:
        labeled_array = result
        num_labels = np.max(labeled_array)

    labeled_array = labeled_array.reshape((-1,))

    bodies = {}
    for label_idx in range(1, num_labels + 1):
        label_vertices = vertices[labeled_array == label_idx]
        body_mesh = m.Mesh(
            np.zeros(label_vertices.shape[0], dtype=m.Mesh.dtype))
        for i, vertex in enumerate(label_vertices):
            body_mesh.vectors[i] = vertex
        bodies[f'{output_base_name}{label_idx}'] = body_mesh

    print(f'Number of bodies extracted: {len(bodies)}')
    for body_name, body_mesh in bodies.items():
        output_filename = f'{output_dir}/{body_name}.stl'
        body_mesh.save(output_filename)
        print(f'Saved {body_name} to {output_filename}')

    return bodies


force_remove_all("for_presentation/out")
get_stl_info("data/stl/cube.stl","for_presentation/out")