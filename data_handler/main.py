from tools import *


# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"
obj_output_dir = "output/CAD/obj"
stl_output_dir = "output/CAD/stl"
info_output_dir = "output/infos"
video_output_dir = "output/video"

try:
    os.mkdir(body_output_dir)
    os.mkdir(dwg_output_dir)
    os.mkdir(image_output_dir)
    os.mkdir(obj_output_dir)
    os.mkdir(stl_output_dir)
    os.mkdir(info_output_dir)
except Exception:
    pass

## local functions
def force_remove_all(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"Failed to remove {item_path}: {e}")

    print(f"All files and directories in {directory_path} have been removed.")

def clear_output_paths():
    force_remove_all(body_output_dir)
    force_remove_all(dwg_output_dir)
    force_remove_all(image_output_dir)
    force_remove_all(obj_output_dir)
    force_remove_all(stl_output_dir)
    force_remove_all(info_output_dir)
    force_remove_all(video_output_dir)


# global functions
def removeUnneededlines(path_to_dwg:str,output_path:str,triangleErrorRange:float = 0,output_name:str="fixed_dwg"):
    doc = ezdxf.readfile(path_to_dwg)
    msp = doc.modelspace()
    t = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]

    tri = []
    for i in range(len(t)):
        p = t[i].get_points()
        if len(p) == 4:
            p.pop()
        for j in range(len(p)):
            p[j] = p[j][:2]
        tri.append(p)
    
    invTri = []
    for i in rmSame(tri):
        if not is_valid_triangle(i,triangleErrorRange):
            invTri.append(i)

    newDoc = ezdxf.new()
    newMSP = newDoc.modelspace()
    for i in invTri:
        newMSP.add_lwpolyline(i)
    
    print(f"created new modulespace with {len(invTri)} lines")
    newDoc.saveas(os.path.join(output_path,f"{output_name}.dwg"))

def get_dwg_info(path_to_dwg:str,output_path:str,outputfoldername:str = "dwg_file_infos"):
    output_path = os.path.join(output_path,outputfoldername)
    os.mkdir(output_path)
    out = view_dwg(path_to_dwg,output_path,"dwg.png",None,True)
    bounding = out[1]["bounding_box"]
    return_dwg_parts(path_to_dwg,output_path)

    t = []

    files = os.listdir(os.path.join(output_path,"dwgparts"))
    os.makedirs(os.path.join(output_path,"dwgpartsimage"))
    for i in range(len(files)):
        print(f"{i+1}/{len(files)} dwg image loaded")
        file_path = os.path.join(os.path.join(output_path,"dwgparts"), files[i])
        if os.path.isfile(file_path):
            view_dwg(file_path,os.path.join(output_path,"dwgpartsimage"),f"#{i+1}triangle",bounding)
            tmpdoc = ezdxf.readfile(file_path)
            tmpmsp = [entity for entity in tmpdoc.modelspace() if entity.dxftype() == "LWPOLYLINE"]
            print(f"loaded #{i+1} triangle modulespace data: consists of {len(tmpmsp)} LWPOLYLINE")
            for i in tmpmsp:
                t.append(i)

    png_to_mp4(os.path.join(output_path,"dwgpartsimage"),output_path,"dwgpartsmovie")

    doc = ezdxf.readfile(path_to_dwg)
    msp = doc.modelspace()

    triDATA = [[],[],[],[],[]] ## number, valid/invalid, points, true points, area
    s = [entity for entity in msp if entity.dxftype() == "LINE"]
    for i in range(len(t)):
        p = t[i].get_points()
        triDATA[0].append(i+1)
        triDATA[3].append(str(p))
        if len(p) == 4:
            p.pop()
        for j in range(len(p)):
            p[j] = p[j][:2]
        triDATA[2].append(str(p))
        try:
            triDATA[4].append(calculateAreaOfTriangle(p))
        except:
            triDATA[4].append(None)
        try:
            if is_valid_triangle(p):
                triDATA[1].append("valid")
            else:
                triDATA[1].append("invalid")
        except:
            triDATA[1].append("is Line")
    createSimpleXLSX(["#","validity","points","true points","area"],triDATA,output_path,"info")
            

    print("created info file")
    print("done!")

def return_dwg_parts(path_to_dwg: str, outputdir: str, foldername: str = "dwgparts"):
    try:
        doc = ezdxf.readfile(path_to_dwg)
    except IOError:
        print(f"Could not read file: {path_to_dwg}")
        return
    except ezdxf.DXFStructureError:
        print(f"Invalid DXF file: {path_to_dwg}")
        return
    
    msp = doc.modelspace()
    
    folder = os.path.join(outputdir, foldername)
    os.makedirs(folder, exist_ok=True)
    
    poly = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    
    j = 0
    for i, lwpolyline in enumerate(poly):
        j+=1
        print(f"{j+1}/{len(poly)} entites splitted")
        tmpdoc = ezdxf.new()
        tmpmsp = tmpdoc.modelspace()
        
        with lwpolyline.points() as points:
            tmpmsp.add_lwpolyline(points)
        
        tmpdoc.saveas(os.path.join(folder, f"triangle{i+1}.dwg"))
     
def dwg_get_points(dwg_file:str):
    dwgFile = ezdxf.readfile(dwg_file)
    msp = dwgFile.modelspace()

    points = []
    for entity in msp:
        for i in entity.get_points():
            points.append(i)
    
    points = rmSame(points)
    print(points)
    return points

def plt_image_saver(plt,output_dir:str,output_name:str = "image_folder"):
    os.makedirs(f"{output_dir}/{output_name}")

def convert_stl_to_step(stl_path, step_path):
    """
    Convert an STL file to a STEP file.

    Args:
        stl_path (str): Path to the input STL file.
        step_path (str): Path to save the output STEP file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    if not os.path.isfile(stl_path):
        print(f"Error: File '{stl_path}' not found.")
        return False

    try:
        subprocess.run(["meshconv", "-c", "step", stl_path,
                       "-o", step_path], check=True)
        print(f"Conversion successful. STEP file saved at '{step_path}'.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Conversion failed. {e}")
        return False

def view_dwg(dwg_path: str, output_dir: str, output_name: str = "dwg_view.png", 
             start_end_points: tuple = None, return_info: bool = False):
    dwg = ezdxf.readfile(dwg_path)
    msp = dwg.modelspace()
    entities = list(msp)
    
    start_x = end_x = start_y = end_y = None
    # Iterate over all entities to find the bounding box
    if start_end_points:
        start_x, start_y = start_end_points[0]
        end_x, end_y = start_end_points[1]
    else:
        for entity in entities:
            if entity.dxftype() == 'LINE':
                start_x = min(start_x, entity.dxf.start[0]) if start_x is not None else entity.dxf.start[0]
                end_x = max(end_x, entity.dxf.end[0]) if end_x is not None else entity.dxf.end[0]
                start_y = min(start_y, entity.dxf.start[1]) if start_y is not None else entity.dxf.start[1]
                end_y = max(end_y, entity.dxf.end[1]) if end_y is not None else entity.dxf.end[1]
            else:
                # Get all vertices of the entity
                vertices = entity.vertices()
                if vertices:
                    x_vals, y_vals = zip(*[(vertex[0], vertex[1]) for vertex in vertices])
                    start_x = min(start_x, min(x_vals)) if start_x is not None else min(x_vals)
                    end_x = max(end_x, max(x_vals)) if end_x is not None else max(x_vals)
                    start_y = min(start_y, min(y_vals)) if start_y is not None else min(y_vals)
                    end_y = max(end_y, max(y_vals)) if end_y is not None else max(y_vals)
        
        start_x -= (end_x-start_x)/15
        end_x += (end_x-start_x)/15
        start_y -= (end_y-start_y)/15
        end_y += (end_y-start_y)/15


    fig = plt.figure()
    ax = fig.add_subplot(111)

    for entity in entities:
        if entity.dxftype() == 'LINE':
            start_point = (entity.dxf.start[0], entity.dxf.start[1])
            end_point = (entity.dxf.end[0], entity.dxf.end[1])
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b-')
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points()
            x = [point[0] for point in points]  # Extract x-coordinates
            y = [point[1] for point in points]  # Extract y-coordinates
            ax.plot(x, y, 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    file_name, file_dot = os.path.splitext(output_name)
    ax.set_title(file_name)

    # Set plot display area based on start and end points
    if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
        ax.set_xlim(start_x, end_x)
        ax.set_ylim(start_y, end_y)

    plt.show()
    plt.savefig(os.path.join(output_dir, output_name))
    
    if return_info:
        info = {
            'bounding_box': ((start_x, start_y), (end_x, end_y)),
            'num_entities': len(entities),
            'format': dwg.dxfversion,
            'filename': os.path.splitext(os.path.basename(dwg_path))[0]
        }
        return plt, info
    else:
        return plt

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

def slice_stl_to_dwg(stl_path: str, slicing_plane_normal: list, slicing_plane_point:list, output_dir: str, output_base_name: str = "sliced"):
    os.makedirs(output_dir, exist_ok=True)

    dwg = ezdxf.new()
    msp = dwg.modelspace()
    stl_model = m.Mesh.from_file(stl_path)

    line = []
    triangle = []
    n=0
    for triangle in stl_model.vectors:
        out = SliceTriangleAtPlane(np.array(slicing_plane_normal),np.array(slicing_plane_point),triangle)
        n+=1
        if out != None:
            print(n,out)
            print(triangle)
            for i in range(len(out)):
                out[i] = out[i][:-1]
            msp.add_lwpolyline(out)
            
    file_out = os.path.join(output_dir,output_base_name+".dwg")
    dwg.saveas(file_out)

def view_stl(stl_path: str, output_dir: str, output_name: str = "stl_view.png"):
    target_stl = m.Mesh.from_file(stl_path)

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
    ax.set_title('Mesh Visualization')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.show()
    plt.savefig(f"{output_dir}/{output_name}")

    return plt

def obj_to_stl(obj_path: str, output_dir: str, output_name: str = "obj_converted.stl"):
    vertices = []
    faces = []

    # Load vertices and faces from the OBJ file
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                if len(face) != 3:  # Check if face has exactly 3 vertices
                    print(f"Skipping face with invalid number of vertices: {face}")
                else:
                    faces.append(face)

    # Check if any faces were loaded
    if not faces:
        print("No valid faces found in the OBJ file.")
        return

    # Convert vertices and faces to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create an STL mesh
    mesh_data = m.Mesh(np.zeros(len(faces), dtype=m.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[f[j], :]

    # Save the mesh to an STL file
    output_file = os.path.join(output_dir, output_name)
    mesh_data.save(output_file)

def view_obj(obj_path: str, output_dir: str, output_base_name: str = "obj_view.png"):
    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)

    if len(vertices) == 0:
        print("No vertices found in the .obj file.")
        return

    vertices = np.array(vertices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(f"{output_dir}/{output_base_name}")
    plt.show()

    return plt

def slice_obj_file(input_file, slice_thickness, output_dir):
    # Load the OBJ file
    vertices = []
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = [float(v) for v in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('l '):
                edge = [int(v) for v in line.strip().split()[1:]]
                edges.append(edge)

    vertices = np.array(vertices)
    edges = np.array(edges) - 1  # Obj files use 1-based indexing

    # Determine the bounding box of the object
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    bounding_box = max_bound - min_bound
    num_slices = int(bounding_box[2] / slice_thickness)

    # Create slices
    for i in range(num_slices):
        z_min = min_bound[2] + i * slice_thickness
        z_max = z_min + slice_thickness

        # Find edges intersecting the slice
        intersecting_edges = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            if v1[2] <= z_max and v2[2] >= z_min or v2[2] <= z_max and v1[2] >= z_min:
                intersecting_edges.append([v1, v2])

        # Create polygons for the slice
        polygons = []
        for edge in intersecting_edges:
            x1, y1, z1 = edge[0]
            x2, y2, z2 = edge[1]
            # Project the edge onto the XY plane
            if z1 < z_min:
                x1 = x2 - (x2 - x1) * (z2 - z_min) / (z2 - z1)
                y1 = y2 - (y2 - y1) * (z2 - z_min) / (z2 - z1)
                z1 = z_min
            if z2 > z_max:
                x2 = x1 + (x2 - x1) * (z_max - z1) / (z2 - z1)
                y2 = y1 + (y2 - y1) * (z_max - z1) / (z2 - z1)
                z2 = z_max
            polygons.append(Polygon([(x1, y1), (x2, y2)]))

        # Create a DXF file for this slice
        dxf = ezdxf.new()
        msp = dxf.modelspace()

        for polygon in polygons:
            msp.add_lwpolyline(list(polygon.exterior.coords), close=True)

        output_file = os.path.join(output_dir, f"slice_{i}.dwg")
        dxf.saveas(output_file)
        print(f"created sliced obj file to {output_file}")
