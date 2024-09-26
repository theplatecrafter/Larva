import ezdxf.document
import ezdxf.entities
import ezdxf.entities
from .tools import *
from collections import defaultdict

# global function
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



# base slicers

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
            msp[-1].dxf.color = 5
            printIF(printDeets,f"{decomposed} has line data: {decomposed.bounds(slice_2d.vertices)}","slice_3d_stl")

    return doc

def width_slice_stl(input_trimesh:trimesh.Trimesh,sliceWidth:float,slicePlaneNormal:list = [0,0,1],printDeets:bool = False):
    docList = []
    points = np.unique(input_trimesh.vertices.reshape([-1, 3]), axis=0)
    slicePlaneNormal = np.array(slicePlaneNormal)
    out = find_signed_min_max_distances(points,slicePlaneNormal)
    minPoint = out[0]
    maxPoint = out[2]
    printIF(printDeets,f"min point:{minPoint} max point:{maxPoint}")
    a = np.array([maxPoint[i] for i in range(len(maxPoint)) if float(slicePlaneNormal[i]) != 0])
    i = np.array([minPoint[i] for i in range(len(minPoint)) if float(slicePlaneNormal[i]) != 0])
    sliceNumbers = math.ceil(np.linalg.norm(a-i)/sliceWidth)
    slicePlanePoints = [minPoint + (slicePlaneNormal/np.linalg.norm(slicePlaneNormal))*i*sliceWidth+(slicePlaneNormal/np.linalg.norm(slicePlaneNormal))*sliceWidth/2 for i in range(sliceNumbers)]
    for i in range(len(slicePlanePoints)):
        doc = slice_stl(input_trimesh,list(slicePlaneNormal),list(slicePlanePoints[i]),printDeets)
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}","width_slice_stl")

    return docList



# smart slicers

def directional_slice_stl(input_trimesh:trimesh.Trimesh,processingDimentionSize:float = 1,noturn:bool = False,printDeets:bool = True) -> list:
    docX = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[1,0,0],printDeets)]
    docY = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,1,0],printDeets)]
    docZ = [simplify_ezdxf_doc(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,0,1],printDeets)]

    singleX = [i for i in docX if count_components(i) == 1]
    singleY = [i for i in docY if count_components(i) == 1]
    singleZ = [i for i in docZ if count_components(i) == 1]
    
    removedX = [i for i in docX if i not in singleX]
    removedY = [i for i in docY if i not in singleY]
    removedZ = [i for i in docZ if i not in singleZ]


    return singleX + singleY + singleZ ## TODO

def gided_layer_slice_stl(input_trimesh:trimesh.Trimesh,sliceWidth:float,slicePlaneNormal:list=None,printDeets:bool = False,margin:float = 6) -> list:
    if not slicePlaneNormal:
        x = gided_layer_slice_stl(input_trimesh,sliceWidth,[1,0,0],printDeets)
        y = gided_layer_slice_stl(input_trimesh,sliceWidth,[0,1,0],printDeets)
        z = gided_layer_slice_stl(input_trimesh,sliceWidth,[0,0,1],printDeets)
        
        xD = [get_dwg_minmaxP(i,True) for i in x]
        yD = [get_dwg_minmaxP(i,True) for i in y]
        zD = [get_dwg_minmaxP(i,True) for i in z]
        
        xDi = sum([i[0]*i[1] for i in xD])
        yDi = sum([i[0]*i[1] for i in yD])
        zDi = sum([i[0]*i[1] for i in zD])
        
        return min({
                x:xDi,
                y:yDi,
                z:zDi
        })
    else:
        docs = width_slice_stl(input_trimesh,sliceWidth,slicePlaneNormal,printDeets)
        for doc in docs:
            msp = doc.modelspace()
            max,min = get_dwg_minmaxP(doc)
            marg = np.array([sliceWidth+margin,sliceWidth+margin])
            msp.add_lwpolyline([
                list(min+(-(sliceWidth+margin),-(sliceWidth+margin)))+[0,0],
                list((min[0],max[1])+(-(sliceWidth+margin),sliceWidth+margin))+[0,0],
                list(max+(sliceWidth+margin,sliceWidth+margin))+[0,0],
                list((max[0],min[1])+(sliceWidth+margin,-(sliceWidth+margin)))+[0,0]
            ])
            msp[-1].dxf.color = 5
            for i in range((1,0),(0,1),(-1,0),(0,-1)):
                


# dwg packers

def grid_pack_dwg(docs:list,Xcount:int = None,margin:float = 1,addGridLine:bool = False):
    packed = ezdxf.new()
    packedMSP = packed.modelspace()
    docs = [i for i in docs if len(i.modelspace()) != 0]
    
    
    if not Xcount:
        Xcount = math.ceil(math.sqrt(len(docs)))
    
    def add_doc(doc,pos,dim):
        msp = doc.modelspace()
        corner = [dim[1][0],dim[0][1]]
        for i in [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]:
            packedMSP.add_lwpolyline(i.translate(pos[0]-corner[0],pos[1]-corner[1],0))
            packedMSP[-1].dxf.color = i.dxf.color
    
    x = 0
    m = 0
    y = 0
    maxY = 0
    for doc in docs:
        m+=1
        max,min = get_dwg_minmaxP(doc)
        if max[1]-min[1] > maxY:
            maxY = max[1]-min[1]


        add_doc(doc,(x,y),(max,min))
        if m == Xcount:
            y-=maxY+margin
            if addGridLine:
                packedMSP.add_lwpolyline([(margin*-1/2,y+margin/2,0,0),(x+max[0]-min[0]+margin/2,y+margin/2,0,0)])
                packedMSP[-1].dxf.color = 1
            x = 0
            m = 0
            maxY = 0
        else:
            x+=max[0]-min[0]+margin
            if addGridLine:
                packedMSP.add_lwpolyline([(x-margin/2,y+margin/2,0,0),(x-margin/2,y-max[1]+min[1]-margin/2,0,0)])
                packedMSP[-1].dxf.color = 1
                
    if addGridLine:
        packedMSP.delete_entity(packedMSP[-1])
            
    return packed
    
def strip_pack_dwg(docs:list,width = None,noturn:bool = False):
    packed = ezdxf.new()
    packedMSP = packed.modelspace()
    docs = [i for i in docs if len(i.modelspace()) != 0]
    
    def add_doc(doc,pos,dim):
        msp = doc.modelspace()
        corner = [dim[1][0],dim[0][1]]
        for i in [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]:
            packedMSP.add_lwpolyline(i.translate(pos[0]-corner[0],pos[1]-corner[1],0))
            packedMSP[-1].dxf.color = i.dxf.color
    
    
    w = 0
    maxmin = []
    for doc in docs:
        max,min = get_dwg_minmaxP(doc)
        maxmin.append([max,min])
        w += max[0]-min[0]
    
    if not width:
        width = w/2
    
    dimensions = [[i[0][0]-i[1][0],i[0][1]-i[1][1]] for i in maxmin]
    
    height, out = strip_pack(width,dimensions,noturn=noturn)
    for i in out:
        try:
            d = dimensions.index([i.w,i.h])
            add_doc(docs[d],[i.x,i.y],maxmin[d])
        except:
            d = dimensions.index([i.h,i.w])
            docs[d] = rotate_dxf(docs[d])
            maxmin[d] = list(get_dwg_minmaxP(docs[d]))

        add_doc(docs[d],[i.x,i.y],[max,min])
        docs.pop(d)
        maxmin.pop(d)
        dimensions.pop(d)
    
    return packed



# other

def simplify_ezdxf_doc(doc: ezdxf.document.Drawing):
    msp = doc.modelspace()
    
    simpDOC = ezdxf.new()
    simpMSP = simpDOC.modelspace()

    def add_lwpolylinez(line):
        simpMSP.add_lwpolyline([list(line[0]) + list(line[0]), list(line[1]) + list(line[1])])
        simpMSP[-1].dxf.color = 5

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
                        add_lwpolylinez(combined_line)
                        processed_lines.add(line)
                        processed_lines.add(connected_line)
                        improved = True
                        break
                else:
                    add_lwpolylinez(line)
                    processed_lines.add(line)
            else:
                add_lwpolylinez(line)
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

def extract_loops(graph):
    """Extract closed loops from the graph using a DFS-like approach."""
    visited = set()
    loops = []

    def dfs(node, path):
        visited.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, path):
                    return path
            elif neighbor == path[0] and len(path) > 2:
                # Found a closed loop
                loops.append(path.copy())
                return True

        path.pop()
        return False

    for node in graph:
        if node not in visited:
            path = []
            dfs(node, path)

    return loops

def is_loop_inside(outer, inner):
    """Check if the inner loop is completely inside the outer loop."""
    # Extract only the (x, y) coordinates from each point in the loop
    outer_polygon = Polygon([(p[0], p[1]) for p in outer])
    
    for point in inner:
        # Ensure Point is from shapely.geometry
        if not outer_polygon.contains(Point(point[0], point[1])):
            return False
    return True

def filter_bigger_loops(loops):
    """Filter out smaller loops that are contained inside larger loops."""
    bigger_loops = []
    for i, loop in enumerate(loops):
        is_inside = False
        for j, other_loop in enumerate(loops):
            if i != j and is_loop_inside(other_loop, loop):
                is_inside = True
                break
        if not is_inside:
            bigger_loops.append(loop)
    return bigger_loops

def count_components(doc: ezdxf.document.Drawing):
    """
    Count how many components (distinct loops, ignoring smaller loops inside larger ones) are inside the DXF file.
    """
    try:
        # Load the DXF file
        modelspace = doc.modelspace()

        entities = []
        # Collect relevant entities
        for entity in modelspace:
            if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE']:
                entities.append(entity)

        # Create graph of connected points
        graph = create_graph(entities)

        # Extract closed loops (parts)
        loops = extract_loops(graph)

        # Filter out smaller loops inside larger loops
        bigger_loops = filter_bigger_loops(loops)

        return len(bigger_loops)

    except ezdxf.DXFStructureError:
        print("Error: Invalid or corrupted DXF file.")
        return 0

