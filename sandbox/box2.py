import ezdxf.math
import Larva.direct_data_handler as dh
import trimesh
import numpy as np
import ezdxf
from ezdxf.addons import r12export
import os
import Larva.read_terminal as rt

def slice_3d_stl(input_file:str, output_dir:str, slice_height:str,output_name:str = "sliced",printDeets:bool = True):
    """
    Slice a 3D STL file at a specified height and export the result to a DXF file.
    
    :param input_file: Path to the input 3D STL file
    :param output_dir: Path to the dir to save the dxf file
    :param slice_height: Height at which to slice the object
    """

    
    mesh = trimesh.load_mesh(input_file)
    slice_origin = [0, 0, slice_height]
    slice_normal = [0, 0, 1]

    # Slice the mesh
    slice_3d = mesh.section(plane_origin = slice_origin,plane_normal = slice_normal)
    if slice_3d is None or len(slice_3d.entities) == 0:
        dh.printIF(printDeets,f"No intersection at height {slice_height}","slice_3d_stl")
        return
    
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    slice_2d = slice_3d.to_planar()[0]
    slice_2d.simplify()
    
    
    for entity in slice_2d.entities:
        for decomposed in entity.explode():
            line = decomposed.bounds(slice_2d.vertices)
            msp.add_lwpolyline([tuple(row) for row in line])
            dh.printIF(printDeets,f"{decomposed} has line data: {decomposed.bounds(slice_2d.vertices)}","slice_3d_stl")


    # Save the dxf file
    output_file = os.path.join(output_dir,output_name+".dxf")
    doc.saveas(output_file)
    dh.printIF(printDeets,f"Slice at height {slice_height} saved to {output_file}","slice_3d_stl")


def create_multiple_slices(input_file, output_dir, slice_width,output_dir_name = "sliced",printDeets:bool = True):
    """
    Create multiple slices of a 3D STL file and save them as separate dxf files.
    
    :param input_file: Path to the input 3D STL file
    :param output_folder: Folder to save the output dxf files
    :param slice_width: Width of each slice
    """
    # Load the STL file
    mesh = trimesh.load_mesh(input_file)

    # Get the bounding box of the mesh
    bounds = mesh.bounds
    zmin, zmax = bounds[:, 2]


    output_folder = os.path.join(output_dir,output_dir_name)
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate number of slices
    num_slices = int((zmax - zmin) / slice_width) + 1

    # Create slices
    n = 0
    for i in range(num_slices):
        n+=1
        slice_height = zmin + i * slice_width
        slice_3d_stl(input_file, output_folder, slice_height,f"slice_{i:03d}",printDeets)

    dh.printIF(printDeets,f"{n}/{num_slices}: Created {num_slices} slices in {output_folder}","create_multiple_slices")

def combine_dxf_files(input_folder, output_file, printDeets: bool = True):
    """
    Combine multiple dxf files into a single dxf file, without overlapping.
    
    :param input_folder: Path to the folder containing input dxf files
    :param output_file: Path to save the combined output dxf file
    :param printDeets: Boolean to control printing of details
    """
    # Create a new dxf file
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Get all dxf files in the input folder
    dxf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.dxf')]

    # Variables to track the overall layout
    current_x, current_y = 0, 0
    max_height_in_row = 0
    max_width = 0

    for filename in dxf_files:
        filepath = os.path.join(input_folder, filename)
        src_doc = ezdxf.readfile(filepath)
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
        if current_x + width > max_width and current_x > 0:
            current_x = 0
            current_y += max_height_in_row
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
        current_x += width
        max_height_in_row = max(max_height_in_row, height)
        max_width = max(max_width, current_x)

    # Save the combined dxf file
    doc.saveas(output_file)
    doc = ezdxf.readfile(output_file)
    r12export.saveas(doc,output_file)

    dh.printIF(printDeets,f"Combined dxf file saved as {output_file}","combine_dxf_files")
    dh.printIF(printDeets,f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})","combine_dxf_files")
    dh.printIF(printDeets,f"Total width: {max_x - min_x}, Total height: {max_y - min_y}","combine_dxf_files")



stl = "data/stl/hollow_cube.stl"



dh.clear_output_paths()
##dh.DEVview_stl(stl,dh.image_output_dir)
create_multiple_slices(stl, dh.slices_output_dir,1,printDeets=False)
dh.view_dwg("output/slices/sliced/slice_005.dxf",dh.image_output_dir)

