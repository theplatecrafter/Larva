from tools import *


# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"
obj_output_dir = "output/CAD/obj"
stl_output_dir = "output/CAD/stl"

try:
    os.mkdir(body_output_dir)
    os.mkdir(dwg_output_dir)
    os.mkdir(image_output_dir)
    os.mkdir(obj_output_dir)
    os.mkdir(stl_output_dir)
except Exception:
    pass

# local functions
def get_dwg_info(path_to_dwg:str,output_path:str):
    output_path = os.path.join(output_path,"dwg_file_infos")
    os.mkdir(output_path)
    out = view_dwg(path_to_dwg,output_path,"dwg.png",None,True)
    bounding = out[1]["bounding_box"]
    return_dwg_parts(path_to_dwg,output_path)

    files = os.listdir(os.path.join(output_path,"dwgparts"))
    os.makedirs(os.path.join(output_path,"dwgpartsimage"))
    for i in range(len(files)):
        print(f"{i+1}/{len(files)} dwg image loaded")
        file_path = os.path.join(os.path.join(output_path,"dwgparts"), files[i])
        if os.path.isfile(file_path):
            view_dwg(file_path,os.path.join(output_path,"dwgpartsimage"),f"#{i+1}triangle",bounding)

    png_to_mp4(os.path.join(output_path,"dwgpartsimage"),output_path,"dwgpartsmovie.mp4",int(math.log(len(files)+1,1.3)))

    txtinfo = open(os.path.join(output_path,"info.txt"),"w")
    txtinfo.write(f"dwg file path: {path_to_dwg}\n")
    doc = ezdxf.readfile(path_to_dwg)
    msp = doc.modelspace()
    t = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    s = [entity for entity in msp if entity.dxftype() == "LINE"]
    txtinfo.write(f"{len(t)} triangles, {len(s)} lines\n\ntriangles:\n")
    for i in range(len(t)):
        p = t[i].get_points()
        if len(p) == 4:
            p.pop()
        for j in range(len(p)):
            p[j] = p[j][:2]
        txtinfo.write(f"triangle {i+1}: {p}\n")
    
    txtinfo.write("\n\nlines:\n")

    for i in range(len(s)):
        p = s[i].get_points()
        for j in range(len(p)):
            p[j] = p[j][:2]
        txtinfo.write(f"line {i+1}: {p}\n")

    lines = []
    for entity in t:
        points = entity.get_points()
        for i in range(len(points)):
            for j in range(len(points)):
                lines.append([(points[i][0],points[i][1]),(points[j][0],points[j][1])])
    
    newLines = []
    for i in range(len(lines)):
        if lines[i][0] != lines[i][1]:
            newLines.append(lines[i])
    lines = rmSame(newLines)

    newLines = []
    for i in range(len(lines)):
        checkFor = [lines[i][1],lines[i][0]]
        if checkFor not in newLines:
            newLines.append(lines[i])
    lines = newLines
    
    txtinfo.write("\n\nlines that can be made with this file:\n")
    for i in range(len(lines)):
        txtinfo.write(f"line{i+1}: {lines[i]}\n")

    allpoints = []
    x,y = [],[]
    for entity in msp:
        p = entity.get_points()
        for i in p:
            allpoints.append(i[:2])
            x.append(allpoints[-1][0])
            y.append(allpoints[-1][1])
        
    allpoints = rmSame(allpoints)
    txtinfo.write(f"\n\ncenter:({(max(x)+min(x))/2},{(max(y)+min(y))/2})\n")
    txtinfo.write(f"dimension: {max(x)-min(x)} x {max(y)-min(y)}\n")
    txtinfo.write(f"\n\nall points ({len(allpoints)} points exists):\n")
    for i in range(len(allpoints)):
        txtinfo.write(f"point {i+1}: {allpoints[i]}\n")
    txtinfo.close()

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
    try:
        force_remove_all("output/dwg_file_infos")
        os.rmdir("output/dwg_file_infos")
    except Exception:
        pass

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

def slice_stl_to_dwg(stl_path: str, slicing_planes: list, output_dir: str, output_base_name: str = "sliced_stl_"):
    os.makedirs(output_dir, exist_ok=True)

    stl_model = m.Mesh.from_file(stl_path)

    n = 0
    for plane_index, (axis, position) in enumerate(slicing_planes):
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        direction = np.sign(position)

        intersecting_polygons = []
        for triangle in stl_model.vectors:
            if (triangle[:, axis_index].min() * direction <= position * direction <= triangle[:, axis_index].max() * direction):
                intersecting_polygons.append(triangle)

        dxf = ezdxf.new("R2010")
        msp = dxf.modelspace()

        for polygon in intersecting_polygons:

            points = polygon[:, [0, 1]]
            points = np.vstack([points, points[0]])
            msp.add_lwpolyline(points)
        n += 1
        output_filename = f"{output_base_name}{n}.dwg"
        output_path = os.path.join(output_dir, output_filename)
        dxf.saveas(output_path)
        print(f"Saved {len(intersecting_polygons)} polygons to {output_path}")

def view_stl(stl_path: str, output_dir: str, output_name: str = "stl_view.png"):
    target_stl = m.Mesh.from_file(stl_path)

    min_x, max_x = np.min(target_stl.vectors[:, :, 0]), np.max(
        target_stl.vectors[:, :, 0])
    min_y, max_y = np.min(target_stl.vectors[:, :, 1]), np.max(
        target_stl.vectors[:, :, 1])
    min_z, max_z = np.min(target_stl.vectors[:, :, 2]), np.max(
        target_stl.vectors[:, :, 2])

    length_x = max_x - min_x
    length_y = max_y - min_y
    length_z = max_z - min_z

    largest_length = max(length_x, length_y, length_z)

    margin = largest_length * 0.1

    xlim = [(min_x + max_x - largest_length) / 2 - margin,
            (min_x + max_x + largest_length) / 2 + margin]
    ylim = [(min_y + max_y - largest_length) / 2 - margin,
            (min_y + max_y + largest_length) / 2 + margin]
    zlim = [(min_z + max_z - largest_length) / 2 - margin,
            (min_z + max_z + largest_length) / 2 + margin]

    x = 1600
    y = 1200
    fig = plt.figure(figsize=(x/100, y/100), dpi=100)

    ax = fig.add_subplot(111, projection='3d')

    polygons = []

    for i in range(len(target_stl.vectors)):
        tri = target_stl.vectors[i]
        polygons.append(tri)

    ax.add_collection3d(Poly3DCollection(
        polygons, color='cyan', linewidths=1, edgecolors='blue'))

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