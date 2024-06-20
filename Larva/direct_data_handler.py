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


## global functions

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
    
    newDoc.saveas(os.path.join(output_path,output_name+".dwg"))
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

def pack_dwg_files_no_overlap(docs):
    combined_doc = ezdxf.new()
    combined_msp = combined_doc.modelspace()
    current_position = (0, 0, 0)
    
    for doc in docs:
        entities = doc.modelspace()
        
        min_point = [float('inf'), float('inf'), float('inf')]
        max_point = [-float('inf'), -float('inf'), -float('inf')]
        
        # Find extents of all entities in the document
        for entity in entities:
            entity_min, entity_max = get_entity_extents(entity)
            
            min_point[0] = min(min_point[0], entity_min[0])
            min_point[1] = min(min_point[1], entity_min[1])
            min_point[2] = min(min_point[2], entity_min[2])
            
            max_point[0] = max(max_point[0], entity_max[0])
            max_point[1] = max(max_point[1], entity_max[1])
            max_point[2] = max(max_point[2], entity_max[2])
        
        # Calculate translation vector
        translation = (
            current_position[0] - min_point[0],
            current_position[1] - min_point[1],
            current_position[2] - min_point[2]
        )
        
        # Translate entities in the DWG file and collect new entities
        new_entities = translate_entities(entities, translation)
        
        # Add new entities to the combined document
        for new_entity in new_entities:
            combined_msp.add_entity(new_entity)
        
        # Update current_position for the next DWG
        current_position = (
            max_point[0] + 10,  # Adding 10 units as a gap to avoid overlap
            current_position[1],
            current_position[2]
        )
    
    return combined_doc
