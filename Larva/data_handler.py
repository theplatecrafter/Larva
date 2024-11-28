from .tools import *
from collections import defaultdict

## global variable
output_version = "AC1032"




# global function
def return_dwg_parts(doc,printDeets:bool = False):
    msp = doc.modelspace()
    
    poly = [entity for entity in msp if entity.dxftype() == "LWPOLYLINE"]
    
    j = 0
    docs = []
    for i, lwpolyline in enumerate(poly):
        j+=1
        printIF(printDeets,f"{j+1}/{len(poly)} entites splitted","return_dwg_parts")
        tmpdoc = ezdxf.new(output_version)
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
    
    doc = ezdxf.new(output_version)
    msp = doc.modelspace()

    slice_2d = slice_3d.to_planar()[0]
    slice_2d.simplify()
    
    
    for entity in slice_2d.entities:
        for decomposed in entity.explode():
            line = decomposed.bounds(slice_2d.vertices)
            msp.add_line((line[0]),list(line[1]))
            msp[-1].dxf.color = 5
            printIF(printDeets,f"{decomposed} has line data: {decomposed.bounds(slice_2d.vertices)}","slice_3d_stl")

    return doc

def width_slice_stl(input_trimesh:trimesh.Trimesh,sliceWidth:float,slicePlaneNormal:list = [0,0,1],printDeets:bool = False):
    docList = []
    points = np.unique(input_trimesh.vertices.reshape([-1, 3]), axis=0)
    slicePlaneNormal = np.array(slicePlaneNormal)/np.linalg.norm(slicePlaneNormal)
    out = find_signed_min_max_distances(points,slicePlaneNormal)
    minPoint = out[0]
    maxPoint = out[2]
    printIF(printDeets,f"min point:{minPoint} max point:{maxPoint}","width_slice_stl")
    a = []
    b = []
    for i in range(len(slicePlaneNormal)):
        if float(slicePlaneNormal[i]) == 0:
            a.append(0)
            b.append(0)
        else:
            a.append(maxPoint[i])
            b.append(minPoint[i])
    a, b = np.array(a), np.array(b)
    sliceNumbers = math.ceil(np.linalg.norm(a-b)/sliceWidth)
    slicePlanePoints = [b + slicePlaneNormal*i*sliceWidth+slicePlaneNormal*sliceWidth/2 for i in range(sliceNumbers)]
    for i in range(len(slicePlanePoints)):
        doc = slice_stl(input_trimesh,list(slicePlaneNormal),list(slicePlanePoints[i]),printDeets)
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}","width_slice_stl")

    return [i for i in docList if i]



# smart slicers

def directional_slice_stl(input_trimesh:trimesh.Trimesh,processingDimentionSize:float = 1,noturn:bool = False,printDeets:bool = True,tolerance:float=1e-9) -> list:
    docX = [simplify_drawing(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[1,0,0],printDeets)]
    docY = [simplify_drawing(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,1,0],printDeets)]
    docZ = [simplify_drawing(i) for i in width_slice_stl(input_trimesh,processingDimentionSize,[0,0,1],printDeets)]

    singleX = [i for i in docX if count_inside_parts(i) == 1]
    singleY = [i for i in docY if count_inside_parts(i) == 1]
    singleZ = [i for i in docZ if count_inside_parts(i) == 1]
    
    removedX = [i for i in docX if i not in singleX]
    removedY = [i for i in docY if i not in singleY]
    removedZ = [i for i in docZ if i not in singleZ]

    return singleX + singleY + singleZ ## TODO

def guided_layer_slice_stl(input_trimesh:trimesh.Trimesh,sliceWidth:float,slicePlaneNormal:list=None,printDeets:bool = False,margin:float = 6) -> list:
    if not slicePlaneNormal:
        x = guided_layer_slice_stl(input_trimesh,sliceWidth,[1,0,0],printDeets)
        y = guided_layer_slice_stl(input_trimesh,sliceWidth,[0,1,0],printDeets)
        z = guided_layer_slice_stl(input_trimesh,sliceWidth,[0,0,1],printDeets)
        
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
        docs = [simplify_drawing(i) for i in width_slice_stl(input_trimesh,sliceWidth,slicePlaneNormal,printDeets)]
        max, min = [], []
        dimX, dimY = 0, 0
        for doc in docs:
            maxTMP,minTMP = get_dwg_minmaxP(doc)
            dimXTMP, dimYTMP = get_dwg_minmaxP(doc,True)
            if dimXTMP > dimX:
                dimX = dimXTMP
            if dimYTMP > dimY:
                dimY = dimYTMP
            max.append(maxTMP)
            min.append(minTMP)
        mx = max[0]
        mn = min[0]
        for i in max:
            if i[0] > mx[0]:
                mx[0] = i[0]
            if i[1] > mx[1]:
                mx[1] = i[1]
        for i in min:
            if i[0] > mn[0]:
                mn[0] = i[0]
            if i[1] > mn[1]:
                mn[1] = i[1]
        
        for i in range(len(docs)):
            msp = docs[i].modelspace()
            # Adding first rectangular boundary
            msp.add_line([mn[0] - margin - sliceWidth, mn[1] - margin - sliceWidth], [mn[0] - margin - sliceWidth, mx[1] + margin + sliceWidth])
            msp.add_line([mn[0] - margin - sliceWidth, mx[1] + margin + sliceWidth], [mx[0] + margin + sliceWidth, mx[1] + margin + sliceWidth])
            msp.add_line([mx[0] + margin + sliceWidth, mx[1] + margin + sliceWidth], [mx[0] + margin + sliceWidth, mn[1] - margin - sliceWidth])
            msp.add_line([mx[0] + margin + sliceWidth, mn[1] - margin - sliceWidth], [mn[0] - margin - sliceWidth, mn[1] - margin - sliceWidth])
            msp[-4].dxf.color = 5
            msp[-3].dxf.color = 5
            msp[-2].dxf.color = 5
            msp[-1].dxf.color = 5

            # Adding the first inner rectangle
            start_point = [mn[0] - margin / 2 - sliceWidth, mn[1] + dimY / 2 - dimY / 4]
            msp.add_line(start_point, [start_point[0], start_point[1] + dimY / 2])
            msp.add_line([start_point[0], start_point[1] + dimY / 2], [start_point[0] + sliceWidth, start_point[1] + dimY / 2])
            msp.add_line([start_point[0] + sliceWidth, start_point[1] + dimY / 2], [start_point[0] + sliceWidth, start_point[1]])
            msp.add_line([start_point[0] + sliceWidth, start_point[1]], start_point)
            msp[-4].dxf.color = 5
            msp[-3].dxf.color = 5
            msp[-2].dxf.color = 5
            msp[-1].dxf.color = 5

            # Adding the second inner rectangle
            start_point = [mn[0] + dimX / 2 - dimX / 4, mx[1] + margin / 2]
            msp.add_line(start_point, [start_point[0] + dimX / 2, start_point[1]])
            msp.add_line([start_point[0] + dimX / 2, start_point[1]], [start_point[0] + dimX / 2, start_point[1] + sliceWidth])
            msp.add_line([start_point[0] + dimX / 2, start_point[1] + sliceWidth], [start_point[0], start_point[1] + sliceWidth])
            msp.add_line([start_point[0], start_point[1] + sliceWidth], start_point)
            msp[-4].dxf.color = 5
            msp[-3].dxf.color = 5
            msp[-2].dxf.color = 5
            msp[-1].dxf.color = 5

            # Adding the third inner rectangle
            start_point = [mx[0] + margin / 2, mn[1] + dimY / 2 - dimY / 4]
            msp.add_line(start_point, [start_point[0], start_point[1] + dimY / 2])
            msp.add_line([start_point[0], start_point[1] + dimY / 2], [start_point[0] + sliceWidth, start_point[1] + dimY / 2])
            msp.add_line([start_point[0] + sliceWidth, start_point[1] + dimY / 2], [start_point[0] + sliceWidth, start_point[1]])
            msp.add_line([start_point[0] + sliceWidth, start_point[1]], start_point)
            msp[-4].dxf.color = 5
            msp[-3].dxf.color = 5
            msp[-2].dxf.color = 5
            msp[-1].dxf.color = 5

            # Adding the fourth inner rectangle
            start_point = [mn[0] + dimX / 2 - dimX / 4, mn[1] - margin / 2 - sliceWidth]
            msp.add_line(start_point, [start_point[0] + dimX / 2, start_point[1]])
            msp.add_line([start_point[0] + dimX / 2, start_point[1]], [start_point[0] + dimX / 2, start_point[1] + sliceWidth])
            msp.add_line([start_point[0] + dimX / 2, start_point[1] + sliceWidth], [start_point[0], start_point[1] + sliceWidth])
            msp.add_line([start_point[0], start_point[1] + sliceWidth], start_point)
            msp[-4].dxf.color = 5
            msp[-3].dxf.color = 5
            msp[-2].dxf.color = 5
            msp[-1].dxf.color = 5
        
        # Create stablizerX
        stablizerX = ezdxf.new(output_version)
        stablizerXMSP = stablizerX.modelspace()

        # Add four lines instead of lwpolyline for stablizerX
        stablizerXMSP.add_line([0, 0], [sliceWidth * (len(docs) + 1), 0])
        stablizerXMSP.add_line([sliceWidth * (len(docs) + 1), 0], [sliceWidth * (len(docs) + 1), dimX / 2])
        stablizerXMSP.add_line([sliceWidth * (len(docs) + 1), dimX / 2], [0, dimX / 2])
        stablizerXMSP.add_line([0, dimX / 2], [0, 0])
        stablizerXMSP[-4].dxf.color = 5
        stablizerXMSP[-3].dxf.color = 5
        stablizerXMSP[-2].dxf.color = 5
        stablizerXMSP[-1].dxf.color = 5

        # Create stablizerY
        stablizerY = ezdxf.new(output_version)
        stablizerYMSP = stablizerY.modelspace()

        # Add four lines instead of lwpolyline for stablizerY
        stablizerYMSP.add_line([0, 0], [sliceWidth * (len(docs) + 1), 0])
        stablizerYMSP.add_line([sliceWidth * (len(docs) + 1), 0], [sliceWidth * (len(docs) + 1), dimY / 2])
        stablizerYMSP.add_line([sliceWidth * (len(docs) + 1), dimY / 2], [0, dimY / 2])
        stablizerYMSP.add_line([0, dimY / 2], [0, 0])
        stablizerYMSP[-4].dxf.color = 5
        stablizerYMSP[-3].dxf.color = 5
        stablizerYMSP[-2].dxf.color = 5
        stablizerYMSP[-1].dxf.color = 5
        
        docs.append(stablizerX)
        docs.append(stablizerX)
        docs.append(stablizerY)
        docs.append(stablizerY)
        
        return docs





# dwg packers

def grid_pack_dwg(docs: list, Xcount: int = None, margin: float = 1, addGridLine: bool = False):
    packed = ezdxf.new(output_version)
    packedMSP = packed.modelspace()
    docs = [i for i in docs if len(i.modelspace()) != 0]
    
    if not Xcount:
        Xcount = math.ceil(math.sqrt(len(docs)))
    
    def add_doc(doc, pos, dim):
        msp = doc.modelspace()
        corner = [dim[1][0], dim[0][1]]
        for i in [entity for entity in msp if entity.dxftype() == "LINE"]:
            packedMSP.add_line(
                (i.dxf.start.x + pos[0] - corner[0], i.dxf.start.y + pos[1] - corner[1]),
                (i.dxf.end.x + pos[0] - corner[0], i.dxf.end.y + pos[1] - corner[1])
            )
            packedMSP[-1].dxf.color = i.dxf.color
    
    x = 0
    m = 0
    y = 0
    maxY = 0
    for doc in docs:
        m += 1
        max, min = get_dwg_minmaxP(doc)
        if max[1] - min[1] > maxY:
            maxY = max[1] - min[1]

        add_doc(doc, (x, y), (max, min))
        if m == Xcount:
            y -= maxY + margin
            if addGridLine:
                packedMSP.add_line((margin * -1 / 2, y + margin / 2),(x + max[0] - min[0] + margin / 2, y + margin / 2))
                packedMSP[-1].dxf.color = 1
            x = 0
            m = 0
            maxY = 0
        else:
            x += max[0] - min[0] + margin
            if addGridLine:
                packedMSP.add_line((x - margin / 2, y + margin / 2),(x - margin / 2, y - max[1] + min[1] - margin / 2))
                packedMSP[-1].dxf.color = 1
                
    if addGridLine:
        packedMSP.delete_entity(packedMSP[-1])
    
    return packed
    
def strip_pack_dwg(docs: list, width=None, noturn: bool = False):
    packed = ezdxf.new(output_version)
    packedMSP = packed.modelspace()
    docs = [i for i in docs if len(i.modelspace()) != 0]
    
    def add_doc(doc, pos, dim):
        msp = doc.modelspace()
        corner = [dim[1][0], dim[0][1]]
        for i in [entity for entity in msp if entity.dxftype() == "LINE"]:
            packedMSP.add_line(
                (i.dxf.start.x + pos[0] - corner[0], i.dxf.start.y + pos[1] - corner[1]),
                (i.dxf.end.x + pos[0] - corner[0], i.dxf.end.y + pos[1] - corner[1])
            )
            packedMSP[-1].dxf.color = i.dxf.color
    
    w = 0
    maxmin = []
    for doc in docs:
        max, min = get_dwg_minmaxP(doc)
        maxmin.append([max, min])
        w += max[0] - min[0]
    
    if not width:
        width = w / 2
    
    dimensions = [[i[0][0] - i[1][0], i[0][1] - i[1][1]] for i in maxmin]
    
    height, out = strip_pack(width, dimensions, noturn=noturn)
    for i in out:
        try:
            d = dimensions.index([i.w, i.h])
            add_doc(docs[d], [i.x, i.y], maxmin[d])
        except:
            d = dimensions.index([i.h, i.w])
            docs[d] = rotate_dxf(docs[d])
            maxmin[d] = list(get_dwg_minmaxP(docs[d]))

        add_doc(docs[d], [i.x, i.y], maxmin[d])
        docs.pop(d)
        maxmin.pop(d)
        dimensions.pop(d)
    
    return packed



# other
def simplify_drawing(doc: ezdxf.document.Drawing) -> ezdxf.document.Drawing: ##TODO: check if this work #TODO: nop, darn it
    def are_collinear(line1, line2):
        """Check if two lines are collinear by comparing their direction vectors."""
        vec1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
        vec2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
        return vec1[0] * vec2[1] == vec1[1] * vec2[0]

    def overlap_or_connect(line1, line2):
        """Check if two collinear lines overlap or connect directly."""
        points = sorted([line1[0], line1[1], line2[0], line2[1]], key=lambda p: (p[0], p[1]))
        return points[1] == points[2] or points[1][0] <= points[2][0] and points[0][0] <= points[3][0]

    def merge_lines(line1, line2):
        """Merge two collinear, overlapping/connecting lines into one."""
        return [min(line1[0], line1[1], line2[0], line2[1], key=lambda p: (p[0], p[1])),
                max(line1[0], line1[1], line2[0], line2[1], key=lambda p: (p[0], p[1]))]

    packed = ezdxf.new(doc.dxfversion)
    packedMSP = packed.modelspace()
    lines = [(line.dxf.start, line.dxf.end) for line in doc.modelspace() if line.dxftype() == "LINE"]
    
    if not lines:
        return packed
    
    merged_lines = []
    
    for line in lines:
        line = sorted(line, key=lambda p: (p[0], p[1]))  # Sort start and end for consistency
        merged = False
        
        # Try to merge with existing lines
        for i, merged_line in enumerate(merged_lines):
            if are_collinear(merged_line, line) and overlap_or_connect(merged_line, line):
                merged_lines[i] = merge_lines(merged_line, line)
                merged = True
                break
        
        if not merged:
            merged_lines.append(line)
    
    # Add merged lines to new DXF document
    for line in merged_lines:
        packedMSP.add_line(line[0], line[1])
    
    return packed

def count_inside_parts(doc: ezdxf.document.Drawing) -> int: ## TODO: does not work
    """Counts the number of separated 'inside' parts in a DXF drawing."""
    
    def extract_closed_loops(doc: ezdxf.document.Drawing):
        """Extracts all closed loops (polygons) from the DXF drawing's LINE entities."""
        lines = [(line.dxf.start, line.dxf.end) for line in doc.modelspace() if line.dxftype() == "LINE"]
        # Convert Vec3 to tuples to access (x, y) coordinates
        line_strings = [LineString([(start.x, start.y), (end.x, end.y)]) for start, end in lines]
        
        # Union of all lines to combine into connected segments
        combined_lines = unary_union(line_strings)
        
        # Extract closed loops (as Polygons)
        polygons = []
        
        # Handle MultiLineString or GeometryCollection
        if isinstance(combined_lines, GeometryCollection):
            for geom in combined_lines:
                if isinstance(geom, LineString) and geom.is_ring:
                    polygons.append(Polygon(geom))
        elif isinstance(combined_lines, LineString):
            if combined_lines.is_ring:
                polygons.append(Polygon(combined_lines))
        elif isinstance(combined_lines, list):
            for geom in combined_lines:
                if geom.is_ring:
                    polygons.append(Polygon(geom))

        return polygons

    # Extract polygons (closed loops) from the DXF drawing
    polygons = extract_closed_loops(doc)
    
    # Instead of checking if one polygon is inside another, just return the number of distinct polygons
    inside_count = len(polygons)
    
    return inside_count

def combine_lwpolylines(drawing, tolerance=1e-8):  ## make this faster
    modelspace = drawing.modelspace()
    lwpolylines = list(modelspace.query('LWPOLYLINE'))
    combined_polylines = []
    processed = set()

    def are_points_close(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2)) < tolerance

    def to_vec2(point):
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return Vec2(point[0], point[1])
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return Vec2(point.x, point.y)
        else:
            raise ValueError(f"Cannot convert to Vec2: {point}")

    def combine_polylines(start_polyline):
        combined = list(start_polyline.get_points())
        current_end = to_vec2(combined[-1])
        processed.add(start_polyline.dxf.handle)
        color = start_polyline.dxf.color  # Store the color of the first polyline

        while True:
            found_next = False
            for poly in lwpolylines:
                if poly.dxf.handle in processed:
                    continue

                points = list(poly.get_points())
                if are_points_close(current_end, to_vec2(points[0])):
                    combined.extend(points[1:])
                    current_end = to_vec2(points[-1])
                    processed.add(poly.dxf.handle)
                    found_next = True
                    break
                elif are_points_close(current_end, to_vec2(points[-1])):
                    combined.extend(reversed(points[:-1]))
                    current_end = to_vec2(points[0])
                    processed.add(poly.dxf.handle)
                    found_next = True
                    break

            if not found_next:
                break

        return combined, color

    for polyline in lwpolylines:
        if polyline.dxf.handle in processed:
            continue

        combined, color = combine_polylines(polyline)
        
        # Check if it forms a loop
        if are_points_close(to_vec2(combined[0]), to_vec2(combined[-1])):
            # Remove the last point if it's the same as the first
            combined.pop()
            combined_polylines.append((combined, color))

    # Remove original polylines
    for handle in processed:
        modelspace.delete_entity(drawing.entitydb[handle])

    # Add new combined polylines
    for points, color in combined_polylines:
        new_polyline = modelspace.add_lwpolyline(points, dxfattribs={'closed': True, 'color': color})

    return drawing

def inside_outside_classification(drawing):
    # Extract all closed polylines from the drawing
    polylines = [entity for entity in drawing.modelspace().query('LWPOLYLINE') if entity.is_closed]
    
    # Convert ezdxf polylines to shapely polygons
    polygons = []
    for polyline in polylines:
        # Extract only X and Y coordinates
        points = [(p[0], p[1]) for p in polyline.get_points()]
        if len(points) >= 3:  # A polygon needs at least 3 points
            polygon = Polygon(points)
            if polygon.is_valid:
                polygons.append(polygon)
            else:
                # If the polygon is invalid, try to fix it
                polygon = polygon.buffer(0)
                if isinstance(polygon, Polygon):
                    polygons.append(polygon)
                elif isinstance(polygon, MultiPolygon):
                    polygons.extend(list(polygon.geoms))
    
    # Merge overlapping polygons
    merged_polygons = []
    while polygons:
        current = polygons.pop(0)
        merged = current
        i = 0
        while i < len(polygons):
            if merged.intersects(polygons[i]):
                merged = merged.union(polygons.pop(i))
            else:
                i += 1
        merged_polygons.append(merged)
    
    # Find the bounding box of all polygons
    if merged_polygons:
        min_x = min(p.bounds[0] for p in merged_polygons)
        min_y = min(p.bounds[1] for p in merged_polygons)
        max_x = max(p.bounds[2] for p in merged_polygons)
        max_y = max(p.bounds[3] for p in merged_polygons)
        
        # Create a grid of points
        x = np.linspace(min_x, max_x, 100)
        y = np.linspace(min_y, max_y, 100)
        xx, yy = np.meshgrid(x, y)
        
        # Classify each point
        inside_regions = set()
        for i in range(len(xx.flatten())):
            point = Point(xx.flatten()[i], yy.flatten()[i])
            for j, polygon in enumerate(merged_polygons):
                if polygon.contains(point):
                    inside_regions.add(j)
                    break
        
        # Count the number of separate inside regions
        num_inside_regions = len(inside_regions)
    else:
        num_inside_regions = 0
    
    return num_inside_regions

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
    outer_polygon = Polygon([(p[0], p[1]) for p in outer])
    
    for point in inner:
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

        modelspace = doc.modelspace()

        entities = []

        for entity in modelspace:
            if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE']:
                entities.append(entity)

        graph = create_graph(entities)


        loops = extract_loops(graph)


        bigger_loops = filter_bigger_loops(loops)

        return len(bigger_loops)

    except ezdxf.DXFStructureError:
        print("Error: Invalid or corrupted DXF file.")
        return 0

