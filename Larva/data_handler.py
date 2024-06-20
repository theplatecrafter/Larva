from .tools import *


# global function
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

def return_dwg_parts(doc):
    msp = doc.modelspace()
    
    poly = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    
    j = 0
    docs = []
    for i, lwpolyline in enumerate(poly):
        j+=1
        print(f"{j+1}/{len(poly)} entites splitted")
        tmpdoc = ezdxf.new()
        tmpmsp = tmpdoc.modelspace()
        
        with lwpolyline.points() as points:
            tmpmsp.add_lwpolyline(points)
        docs.append(tmpdoc)
    
    return docs

def return_stl_parts(original_mesh:m.Mesh):

    num_triangles = len(original_mesh)
    meshes = []
    for i, triangle in enumerate(original_mesh.vectors):
        print(f"{i+1}/{num_triangles} triangles processed")

        # Create a new mesh with a single triangle
        new_mesh = m.Mesh(np.zeros(1, dtype=m.Mesh.dtype))
        new_mesh.vectors[0] = triangle.reshape((3, 3))

        # Save the new mesh as an STL file
        meshes.append(new_mesh)
    
    return meshes

def return_stl_bodies(stl_mesh:m.Mesh):
    vertices = stl_mesh.vectors.reshape((-1, 3))
    result = label(stl_mesh.vectors[:, :, 0])

    if isinstance(result, tuple):
        labeled_array, num_labels = result
    else:
        labeled_array = result
        num_labels = np.max(labeled_array)

    labeled_array = labeled_array.reshape((-1,))

    bodies = []
    for label_idx in range(1, num_labels + 1):
        label_vertices = vertices[labeled_array == label_idx]
        body_mesh = m.Mesh(
            np.zeros(label_vertices.shape[0], dtype=m.Mesh.dtype))
        for i, vertex in enumerate(label_vertices):
            body_mesh.vectors[i] = vertex
        bodies.append(body_mesh)

    return bodies

def dwg_get_points(doc):
    msp = doc.modelspace()

    points = []
    for entity in msp:
        for i in entity.get_points():
            points.append(i)
    
    points = rmSame(points)
    print(points)
    return points

def slice_stl_to_dwg(mesh:m.Mesh, slicing_plane_normal: list, slicing_plane_point:list,printDeets:bool = False):
    dwg = ezdxf.new()
    msp = dwg.modelspace()
    stl_model = align_mesh_to_cutting_plane(mesh,np.array(slicing_plane_normal),np.array(slicing_plane_point))

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
            
    return dwg

def width_slice_stl(stl_model:m.Mesh,sliceWidth:float,slicePlaneNormal:list = [0,0,1],printDeets:bool = False):
    docList = []
    points = np.unique(stl_model.vectors.reshape([-1, 3]), axis=0)
    slicePlaneNormal = np.array(slicePlaneNormal)
    out = find_signed_min_max_distances(points,slicePlaneNormal)
    minPoint = out[0]
    maxPoint = out[2]
    sliceNumbers = math.ceil(np.linalg.norm(maxPoint-minPoint)/sliceWidth)-1
    slicePlanePoints = [minPoint + (slicePlaneNormal/np.linalg.norm(slicePlaneNormal))*i for i in range(sliceNumbers)]
    for i in range(len(slicePlanePoints)):
        doc = removeUnneededlines(slice_stl_to_dwg(stl_model,list(slicePlaneNormal),list(slicePlanePoints[i]),None,None,False,True))
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}")
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: created image file")


    return docList

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
