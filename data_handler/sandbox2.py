from tools import *
import main as dh

def ccw(A, B, C):
    """Check if three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(line1, line2):
    """Check if two line segments intersect."""
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def detect_overlapping_lines(dwg_file_path, output_file_path):
    # Load the DWG file
    doc = ezdxf.readfile(dwg_file_path)
    msp = doc.modelspace()

    lwpolylines = [entity for entity in msp if entity.dxftype() == 'LWPOLYLINE']

    all_lines = []
    for lwpolyline in lwpolylines:
        with lwpolyline.points() as points:
            vertices = list(points)
            line_segments = [(vertices[i], vertices[i+1]) for i in range(len(vertices) - 1)]
            
        # Append line segments to the list
        all_lines.extend(line_segments)
    
    all_lines = rmSame(all_lines)

    # Check for intersection among all line segments
    lines_to_remove = []
    for line1, line2 in combinations(all_lines, 2):
        if intersect(line1, line2):
            lines_to_remove.append(line1)
            lines_to_remove.append(line2)

    lines_to_remove = rmSame(lines_to_remove)

    for i in range(len(lines_to_remove)):
        if lines_to_remove[i] in all_lines or (lines_to_remove[i][1],lines_to_remove[i][0]) in all_lines:
            if lines_to_remove[i] in all_lines:
                all_lines.remove(lines_to_remove[i])
            else:
                all_lines.remove((lines_to_remove[i][1],lines_to_remove[i][0]))

    return all_lines



# Usage example
dwg_file_path = "output/CAD/dwg/sliced_stl_1.dwg"
output_file_path = "output/CAD/dwg/rminter.dwg"
l = detect_overlapping_lines(dwg_file_path, output_file_path)
print(l)
dh.create_dwg_outof_lines(l,output_file_path)
dh.view_dwg("output/CAD/dwg/rminter.dwg",dh.image_output_dir)