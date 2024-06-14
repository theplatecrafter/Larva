from .tools import *


# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"
obj_output_dir = "output/CAD/obj"
stl_output_dir = "output/CAD/stl"
info_output_dir = "output/infos"
video_output_dir = "output/video"
slices_output_dir = "output/slices"

try:
    os.mkdir("output")
    os.mkdir("output/CAD")
    os.mkdir(body_output_dir)
    os.mkdir(dwg_output_dir)
    os.mkdir(image_output_dir)
    os.mkdir(obj_output_dir)
    os.mkdir(stl_output_dir)
    os.mkdir(info_output_dir)
    os.mkdir(video_output_dir)
    os.mkdir(slices_output_dir)
except Exception:
    pass

## local functions
def clear_output_paths():
    force_remove_all(body_output_dir)
    force_remove_all(dwg_output_dir)
    force_remove_all(image_output_dir)
    force_remove_all(obj_output_dir)
    force_remove_all(stl_output_dir)
    force_remove_all(info_output_dir)
    force_remove_all(video_output_dir)
    force_remove_all(slices_output_dir)


# global functions
def removeUnneededlinesFromFile(path_to_dwg:str,output_path:str,triangleErrorRange:float = 0,output_name:str="fixed_dwg"):
    removeUnneededlines(ezdxf.readfile(path_to_dwg),triangleErrorRange).saveas(os.path.join(output_path,output_name+".dwg"))

def removeUnneededlines(doc,triangleErrorRange:float = 0):
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
    
    return newDoc

def get_dwg_info(path_to_dwg:str,output_path:str,outputfoldername:str = "dwg_file_infos"):
    output_path = os.path.join(output_path,outputfoldername)
    os.mkdir(output_path)
    out = view_dwg(path_to_dwg,output_path,"dwg.png",None,True,True)
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

def view_dwg(dwg_path: str, output_dir: str, output_name: str = "dwg_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave:bool = False):
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
        
        start_x -= (end_x - start_x) / 15
        end_x += (end_x - start_x) / 15
        start_y -= (end_y - start_y) / 15
        end_y += (end_y - start_y) / 15

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Generate a colormap
    num_entities = len(entities)
    colors = hsv_to_rgb(np.linspace(0, 1, num_entities).reshape(-1, 1) * np.ones((1, 3)))

    for idx, entity in enumerate(entities):
        color = colors[idx]
        if entity.dxftype() == 'LINE':
            start_point = (entity.dxf.start[0], entity.dxf.start[1])
            end_point = (entity.dxf.end[0], entity.dxf.end[1])
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)
        elif entity.dxftype() == 'LWPOLYLINE':
            points = entity.get_points()
            x = [point[0] for point in points]  # Extract x-coordinates
            y = [point[1] for point in points]  # Extract y-coordinates
            ax.plot(x, y, color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    file_name, file_dot = os.path.splitext(output_name)
    ax.set_title(file_name)

    # Set plot display area based on start and end points
    if start_x is not None and end_x is not None and start_y is not None and end_y is not None:
        ax.set_xlim(start_x, end_x)
        ax.set_ylim(start_y, end_y)

    if not onlySave:
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

def slice_stl_to_dwg(stl_path: str, slicing_plane_normal: list, slicing_plane_point:list, output_dir: str, output_base_name: str = "sliced",printDeets:bool = False,justDOC:bool = False):
    dwg = ezdxf.new()
    msp = dwg.modelspace()
    stl_model = align_mesh_to_cutting_plane(m.Mesh.from_file(stl_path),np.array(slicing_plane_normal),np.array(slicing_plane_point))

    n=0
    for triangle in stl_model.vectors:
        out = SliceTriangleAtPlane(np.array([0,0,1]),np.array([0,0,0]),triangle,printDeets)
        if out != None:
            n+=1
            zPlane = []
            for i in range(len(out)):
                zPlane.append(out[i][-1])
                out[i] = out[i][:-1]
            out = rmSame(out)
            if len(out) != 1:
                if len(out) == 2:
                    out.append(out[0])
                    out.append(out[1])
                elif len(out) == 3:
                    out.append(out[0])
                msp.add_lwpolyline(out)

                printIF(printDeets,f"{n}: Created LWPOLYLINE : {out}\ntriangle:{triangle}")
            
    if not justDOC:
        os.makedirs(output_dir, exist_ok=True)
        file_out = os.path.join(output_dir,output_base_name+".dwg")
        dwg.saveas(file_out)
    return dwg

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

def width_slice_stl(stl_path:str,outputFolder:str,sliceWidth:float,slicePlaneNormal:list = [0,0,1],outputFolderName:str = "stl_slices",printDeets:bool = False,withExtra:bool = False):
    outFolder = os.path.join(outputFolder,outputFolderName)
    os.mkdir(outFolder)
    CADFolder = os.path.join(outFolder,"slicesDATA")
    os.mkdir(CADFolder)
    if withExtra:
        IMGFolder = os.path.join(outFolder,"sliceIMG")
        os.mkdir(IMGFolder)

    docList = []
    stl_model = m.Mesh.from_file(stl_path)
    points = np.unique(stl_model.vectors.reshape([-1, 3]), axis=0)
    slicePlaneNormal = np.array(slicePlaneNormal)
    out = find_signed_min_max_distances(points,slicePlaneNormal)
    minPoint = out[0]
    maxPoint = out[2]
    sliceNumbers = math.ceil(np.linalg.norm(maxPoint-minPoint)/sliceWidth)-1
    slicePlanePoints = [minPoint + (slicePlaneNormal/np.linalg.norm(slicePlaneNormal))*i for i in range(sliceNumbers)]
    for i in range(len(slicePlanePoints)):
        doc = removeUnneededlines(slice_stl_to_dwg(stl_path,list(slicePlaneNormal),list(slicePlanePoints[i]),None,None,False,True))
        doc.saveas(os.path.join(CADFolder,f"slice{i+1}.dwg"))
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}")
        if withExtra:
            view_dwg(os.path.join(CADFolder,f"slice{i+1}.dwg"),IMGFolder,f"slice{i+1}.png")
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: created image file")
    if withExtra:
        png_to_mp4(IMGFolder,outFolder,"slices")

    combinedDOC = pack_dwg_files_no_overlap(docList)
    combinedDOC.saveas(os.path.join(outFolder,"slice_combined.dwg"))
    if withExtra:
        view_dwg(os.path.join(outFolder,"slice_combined.dwg"),outFolder,"slice_combined.png")