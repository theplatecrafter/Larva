import ezdxf.document
from Larva.tools import *
import Larva.direct_data_handler as dh

def slice_3d_stl(input_trimesh:trimesh.Trimesh, slice_normal:list,slice_origin:list ,printDeets:bool = True):
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
            msp.add_line(line[0],line[1])
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
        doc = slice_3d_stl(input_trimesh,list(slicePlaneNormal),list(slicePlanePoints[i]),printDeets)
        docList.append(doc)
        printIF(printDeets,f"{i+1}/{len(slicePlanePoints)}: sliced mesh with plane point: {slicePlanePoints[i]}","width_slice_stl")

    return docList

def combine_dxf_files(docList:list, gap=0, printDeets: bool = True):
    """
    Combine multiple dxf files into a single dxf file, without overlapping, with a specified gap.
    
    :param input_folder: Path to the folder containing input dxf files
    :param output_file: Path to save the combined output dxf file
    :param gap: The gap to leave between each combined file
    :param printDeets: Boolean to control printing of details
    """
    # Create a new dxf file
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Get all dxf files in the input folder
    # Variables to track the overall layout
    current_x, current_y = 0, 0
    max_height_in_row = 0
    max_width = 0

    for src_doc in docList:
        src_msp = src_doc.modelspace()

        # Calculate bounding box for the current file
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        for entity in src_msp:
            if entity.dxftype() == 'LINE':
                start, end = entity.dxf.start, entity.dxf.end
                min_x = min(min_x, start[0], end[0])
                min_y = min(min_y, start[1], end[1])
                max_x = max(max_x, start[0], end[0])
                max_y = max(max_y, start[1], end[1])
            elif entity.dxftype() == 'ARC':
                center, radius = entity.dxf.center, entity.dxf.radius
                min_x = min(min_x, center[0] - radius)
                min_y = min(min_y, center[1] - radius)
                max_x = max(max_x, center[0] + radius)
                max_y = max(max_y, center[1] + radius)

        # Calculate dimensions of the current file
        width, height = max_x - min_x, max_y - min_y

        # Check if we need to move to a new row
        if current_x + width + gap > max_width and current_x > 0:
            current_x = 0
            current_y += max_height_in_row + gap
            max_height_in_row = 0

        # Calculate offset for the current file
        offset_x, offset_y = current_x - min_x, current_y - min_y

        # Add entities to the new document with offset
        for entity in src_msp:
            if entity.dxftype() == 'LINE':
                start, end = entity.dxf.start, entity.dxf.end
                msp.add_line(
                    (start[0] + offset_x, start[1] + offset_y),
                    (end[0] + offset_x, end[1] + offset_y)
                )
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                msp.add_arc(
                    (center[0] + offset_x, center[1] + offset_y),
                    entity.dxf.radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle
                )

        # Update layout tracking variables
        current_x += width + gap
        max_height_in_row = max(max_height_in_row, height)
        max_width = max(max_width, current_x)

    # Save the combined dxf file
    return doc




stl = "/media/hans/142A-A346/Hans_KADAIKENNKYUU/test.stl"



dh.clear_output_paths()
DEVview_stl(stl,dh.image_output_dir)

mesh = trimesh.load_mesh(stl)

combine_dxf_files(width_slice_stl(mesh,4,[0,0,1]),5).saveas(os.path.join("/media/hans/142A-A346/Hans_KADAIKENNKYUU","out.dwg"))
view_dwg(os.path.join("/media/hans/142A-A346/Hans_KADAIKENNKYUU","out.dwg"),dh.image_output_dir)

