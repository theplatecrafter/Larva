import ezdxf.document
import ezdxf.document
import ezdxf.document
import ezdxf.entities
from .tools import *


# global function
def removeUnneededlines(doc,triangleErrorRange:float = 0,printDeets:bool = False):
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
    printIF(printDeets,"extracted triangle datas","removeUnneededlines")
    invTri = []
    for i in rmSame(tri):
        if not is_valid_triangle(i,triangleErrorRange):
            invTri.append(i)

    newDoc = ezdxf.new()
    newMSP = newDoc.modelspace()
    for i in invTri:
        newMSP.add_lwpolyline(i)
    printIF(printDeets,"extracted triangle datas","removeUnneededlines")
    
    return newDoc

def return_dwg_parts(doc,printDeets:bool = False):
    msp = doc.modelspace()
    
    poly = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    
    j = 0
    docs = []
    for i, lwpolyline in enumerate(poly):
        j+=1
        printIF(printDeets,f"{j+1}/{len(poly)} entites splitted","return_dwg_parts")
        tmpdoc = ezdxf.new()
        tmpmsp = tmpdoc.modelspace()
        
        with lwpolyline.points() as points:
            tmpmsp.add_lwpolyline(points)
        docs.append(tmpdoc)
    
    return docs

def slice_stl(input_trimesh:trimesh.Trimesh, slice_normal:list,slice_origin:list ,printDeets:bool = True):
    """
    Slice a 3D STL file at a specified height and export the result to a DXF file.
    
    :param input_file: Path to the input 3D STL file
    :param output_dir: Path to the dir to save the dxf file
    :param slice_height: Height at which to slice the object
    """

    slice_3d = input_trimesh.section(plane_origin = slice_origin,plane_normal = slice_normal)
    if slice_3d is None or len(slice_3d.entities) == 0:
        printIF(printDeets,f"No intersection at {slice_normal}, {slice_origin}","slice_3d_stl")
        return
    
    doc = ezdxf.new()
    msp = doc.modelspace()

    slice_2d = slice_3d.to_planar()[0]
    slice_2d.simplify()
    
    
    for entity in slice_2d.entities:
        for decomposed in entity.explode():
            line = decomposed.bounds(slice_2d.vertices)
            msp.add_lwpolyline((line[0],line[1]),)
            printIF(printDeets,f"{decomposed} has line data: {decomposed.bounds(slice_2d.vertices)}","slice_3d_stl")

    return doc

def width_slice_stl(input_trimesh:trimesh.Trimesh,sliceWidth:float,slicePlaneNormal:list = [0,0,1],printDeets:bool = False):
    docList = []
    points = np.unique(input_trimesh.vertices.reshape([-1, 3]), axis=0)
    slicePlaneNormal = np.array(slicePlaneNormal)
    out = find_signed_min_max_distances(points,slicePlaneNormal)
    minPoint = out[0]
    maxPoint = out[2]
    sliceNumbers = math.ceil(np.linalg.norm(maxPoint-minPoint)/sliceWidth)
    slicePlanePoints = [minPoint + (slicePlaneNormal/np.linalg.norm(slicePlaneNormal))*i*sliceWidth for i in range(sliceNumbers)]
    for i in range(len(slicePlanePoints)):
        doc = slice_stl(input_trimesh,list(slicePlaneNormal),list(slicePlanePoints[i]),printDeets)
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}","width_slice_stl")

    return docList

def pack_dwg_files_no_overlap(docs,printDeets:bool = False):
    combined_doc = ezdxf.new()
    combined_msp = combined_doc.modelspace()
    current_position = (0, 0, 0)
    
    n = 0
    for doc in docs:
        n+=1
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
        printIF(printDeets,f"{n}/{len(docs)} inserted dwg at {translation}","pack_dwg_files_no_overlap")
    
    return combined_doc

def smart_slice_stl(input_trimesh:trimesh.Trimesh,processingDimentionSize:float = 1,printDeets:bool = True):
    docX = width_slice_stl(input_trimesh,processingDimentionSize,[1,0,0],printDeets)
    docY = width_slice_stl(input_trimesh,processingDimentionSize,[0,1,0],printDeets)
    docZ = width_slice_stl(input_trimesh,processingDimentionSize,[0,0,1],printDeets)
    
    print(simplify_ezdxf_doc(docX[0]))

def dwg_components(doc:ezdxf.document.Drawing):
    msp = doc.modelspace()

    G = nx.Graph()

    def add_edge(p1, p2):
        G.add_edge(tuple(p1), tuple(p2))
        print(f"added {p1}, {p2}")

    for entity in msp:
        if entity.dxftype() == 'LINE':
            add_edge(entity.dxf.start, entity.dxf.end)
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            for i in range(8):
                angle = i * 45
                point = Vec2.from_deg_angle(angle) * radius + Vec2(center)
                add_edge(center, point)
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            for angle in range(int(start_angle), int(end_angle) + 45, 45):
                point = Vec2.from_deg_angle(angle) * radius + Vec2(center)
                add_edge(center, point)

    num_components = nx.number_connected_components(G)

    return num_components

def simplify_ezdxf_doc(doc:ezdxf.document.Drawing):
    msp = doc.modelspace()
    
    def points_are_close(p1, p2, tolerance=1e-6):
        return Vec2(p1).distance(Vec2(p2)) < tolerance

    def lines_are_collinear(line1, line2, tolerance=1e-6):
        vec1 = Vec2(line1.dxf.end) - Vec2(line1.dxf.start)
        vec2 = Vec2(line2.dxf.end) - Vec2(line2.dxf.start)
        return abs(vec1.angle_between(vec2)) < tolerance or abs(vec1.angle_between(vec2) - math.pi) < tolerance
    def merge_collinear_lines(line1, line2):
        points = [Vec2(line1.dxf.start), Vec2(line1.dxf.end), Vec2(line2.dxf.start), Vec2(line2.dxf.end)]
        min_point = min(points, key=lambda p: (p.x, p.y))
        max_point = max(points, key=lambda p: (p.x, p.y))
        return min_point, max_point

    lines = [e for e in msp if e.dxftype() == "LINE"]

    lines.sort(key=lambda l: (l.dxf.start[0], l.dxf.start[1]))

    simplified_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]
        merged = False
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            if lines_are_collinear(current_line, next_line):
                # Merge the lines
                start, end = merge_collinear_lines(current_line, next_line)
                current_line.dxf.start = start
                current_line.dxf.end = end
                merged = True
                j += 1
            else:
                break
        if not merged:
            simplified_lines.append(current_line)
        i = j

    for line in lines:
        msp.delete_entity(line)

    for line in simplified_lines:
        msp.add_line(line.dxf.start, line.dxf.end)

    return doc