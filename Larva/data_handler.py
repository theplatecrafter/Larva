import ezdxf.document
import ezdxf.entities
from .tools import *
from collections import defaultdict

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
            msp.add_lwpolyline([list(line[0]) + list(line[0]),list(line[1]) + list(line[1])])
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


def strip_pack_dwg(docs:list,width = 100):
    packed = ezdxf.new()
    packedMSP = packed.modelspace()
    
    def add_doc(doc,pos,dim):
        msp = doc.modelspace()
        corner = [dim[1][0],dim[0][1]]
        for i in [entity for entity in msp if entity.dxftype() == "LWPOLYLNE"]:
            packedMSP.add_lwpolyline(i.translate(pos[0]-corner[0],pos[1]-corner[1],0))
    
    maxmin = []
    for doc in docs:
        msp = doc.modelspace()
        lwpolyline = [i for i in msp if i.dxftype() == "LWPOLYLINE"]
        min = list(list(lwpolyline[0].vertices())[0])
        max = list(list(lwpolyline[0].vertices())[0])
        for line in lwpolyline:
            for point in list(line.vertices()):
                if min[0] > point[0]:
                    min[0] = point[0]
                elif max[0] < point[0]:
                    max[0] = point[0]
                    
                if min[1] > point[1]:
                    min[1] = point[1]
                elif max[1] < point[1]:
                    max[1] = point[1]
        maxmin.append([max,min])
    
    dimensions = [[i[0][0]-i[1][0],i[0][1]-i[1][1]] for i in maxmin]
    
    height, out = strip_pack(width,dimensions)
    for i in out:
        d = dimensions.index([i.w,i.h])
        print(d,maxmin[d])
        add_doc(docs[d],[i.x,i.y],maxmin[d])
        docs.pop(d)
        maxmin.pop(d)
        dimensions.pop(d)
    
    return packed


def smart_slice_stl(input_trimesh:trimesh.Trimesh,processingDimentionSize:float = 1,printDeets:bool = True):
    docX = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[1,0,0],printDeets)]
    docY = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,1,0],printDeets)]
    docZ = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,0,1],printDeets)]

    singleX = [i for i in docX if count_closed_parts(i) == 1]
    singleY = [i for i in docY if count_closed_parts(i) == 1]
    singleZ = [i for i in docZ if count_closed_parts(i) == 1]
    
    removedX = [i for i in docX if i not in singleX]
    removedY = [i for i in docY if i not in singleY]
    removedZ = [i for i in docZ if i not in singleZ]


    return strip_pack_dwg(docX)

def simplify_ezdxf_doc(doc: ezdxf.document.Drawing):
    msp = doc.modelspace()
    
    simpDOC = ezdxf.new()
    simpMSP = simpDOC.modelspace()

    def add_lwpolyline(line):
        simpMSP.add_lwpolyline([list(line[0]) + list(line[0]), list(line[1]) + list(line[1])])

    improved = True
    while improved:
        lines = [tuple(e.vertices()) for e in msp if e.dxftype() == "LWPOLYLINE"]
        connectivityLINES = create_line_connectivity_map(lines)
        processed_lines = set()
        improved = False
        
        for line in lines:
            if line in processed_lines:
                continue

            try:
                connected_lines = [l for l in get_connected_lines(line, connectivityLINES) if l not in processed_lines]
            except IndexError:
                connected_lines = []

            if connected_lines:
                for connected_line in connected_lines:
                    combined_line = combine_lines(line, connected_line, 0)
                    if combined_line:
                        add_lwpolyline(combined_line)
                        processed_lines.add(line)
                        processed_lines.add(connected_line)
                        improved = True
                        break
                else:
                    add_lwpolyline(line)
                    processed_lines.add(line)
            else:
                add_lwpolyline(line)
                processed_lines.add(line)

        if improved:
            msp = simpMSP
            simpDOC = ezdxf.new()
            simpMSP = simpDOC.modelspace()
        else:
            break

    return simpDOC

def create_graph(entities):
    """Create a graph where points are nodes, and connected entities are edges."""
    graph = defaultdict(list)

    for entity in entities:
        if entity.dxftype() == 'LWPOLYLINE' or entity.dxftype() == 'POLYLINE':
            points = entity.get_points()
            for i in range(len(points) - 1):
                graph[tuple(points[i])].append(tuple(points[i + 1]))
                graph[tuple(points[i + 1])].append(tuple(points[i]))
            if entity.is_closed:
                graph[tuple(points[-1])].append(tuple(points[0]))
                graph[tuple(points[0])].append(tuple(points[-1]))
        elif entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            graph[tuple(start)].append(tuple(end))
            graph[tuple(end)].append(tuple(start))
        elif entity.dxftype() == 'CIRCLE':
            # A circle is inherently closed, so we treat it as a single closed part
            center = tuple(entity.dxf.center[:2])
            graph[center].append(center)

    return graph

def find_closed_loops(graph):
    """Find closed loops in the graph using a DFS-like approach."""
    visited = set()
    closed_parts = 0

    def dfs(node, parent, path):
        visited.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node, path):
                    return True
            elif neighbor == path[0] and len(path) > 2:
                # Found a closed loop
                return True

        path.pop()
        return False

    for node in graph:
        if node not in visited:
            path = []
            if dfs(node, None, path):
                closed_parts += 1

    return closed_parts

def count_closed_parts(doc:ezdxf.document.Drawing):
    """
    Count how many closed parts (connected shapes forming closed loops) are inside the DWG file.
    Parameters:
    dwg_file_path (str): Path to the DWG file.
    Returns:
    int: Number of closed parts detected in the DWG file.
    """
    try:
        # Load the DWG file
        modelspace = doc.modelspace()

        entities = []
        # Collect relevant entities
        for entity in modelspace:
            if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE']:
                entities.append(entity)

        # Create graph of connected points
        graph = create_graph(entities)

        # Find and count closed loops (parts)
        return find_closed_loops(graph)


    except ezdxf.DXFStructureError:
        print("Error: Invalid or corrupted DWG file.")
        return 0
