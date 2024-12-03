from .tools import *
from .data_handler import *


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

def get_dwg_info(path_to_dwg:str,output_path:str,outputfoldername:str = "dwg_file_infos",printDeets:bool = True):
    output_path = os.path.join(output_path,outputfoldername)
    os.mkdir(output_path)
    out = view_dwg(path_to_dwg,output_path,"dwg.png",None,True,True)
    bounding = out[1]["bounding_box"]
    DEVreturn_dwg_parts(path_to_dwg,output_path,printDeets=printDeets)

    t = []

    files = os.listdir(os.path.join(output_path,"dwgparts"))
    os.makedirs(os.path.join(output_path,"dwgpartsimage"))
    for i in range(len(files)):
        printIF(printDeets,f"{i+1}/{len(files)} dwg image loaded","get_dwg_info")
        file_path = os.path.join(os.path.join(output_path,"dwgparts"), files[i])
        if os.path.isfile(file_path):
            view_dwg(file_path,os.path.join(output_path,"dwgpartsimage"),f"#{i+1}triangle",bounding)
            tmpdoc = ezdxf.readfile(file_path)
            tmpmsp = [entity for entity in tmpdoc.modelspace() if entity.dxftype() == "LWPOLYLINE"]
            printIF(printDeets,f"loaded #{i+1} triangle modulespace data: consists of {len(tmpmsp)} LWPOLYLINE","get_dwg_info")
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
            p[j] = p[j][:3]
        triDATA[2].append(str(p))
        try:
            triDATA[4].append(calculateAreaOfTriangle(p,printDeets))
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
            

    printIF(printDeets,"created info file","get_dwg_info")
    printIF(printDeets,"done!","get_dwg_info")

def DEVreturn_dwg_parts(path_to_dwg: str, outputdir: str, foldername: str = "dwgparts",printDeets:bool = True):
    out = os.path.join(outputdir,foldername)
    os.mkdir(out)
    data = return_dwg_parts(ezdxf.readfile(path_to_dwg),printDeets)
    for i in range(len(data)):
        data[i].saveas(os.path.join(out,f"part{i+1}.dwg"))
    
def DEVslice_stl(stl_path: str, slicing_plane_normal: list, slicing_plane_point:list, output_dir: str, output_name: str = "sliced",printDeets:bool = True):
    slice_stl(trimesh.load_mesh(stl_path),slicing_plane_normal,slicing_plane_point,printDeets).saveas(os.path.join(output_dir,output_name+".dwg"))

def DEVwidth_slice_stl(stl_path:str,outputFolder:str,sliceWidth:float,slicePlaneNormal:list = [0,0,1],outputFolderName:str = "stl_slices",printDeets:bool = False,withExtra:bool = True):
    out = width_slice_stl(trimesh.load_mesh(stl_path),sliceWidth,slicePlaneNormal,printDeets)
    path = os.path.join(outputFolder,outputFolderName)
    os.mkdir(path)
    if withExtra:
        os.mkdir(os.path.join(path,"image"))
        os.mkdir(os.path.join(path,"data"))
    for i in range(len(out)):
        if withExtra:
            out[i].saveas(os.path.join(path,os.path.join("data",f"slice{i+1}.dwg")))
            view_dwg(os.path.join(path,os.path.join("data",f"slice{i+1}.dwg")),os.path.join(path,"image"),f"slice{i+1}.png",onlySave=True)
            printIF(printDeets,f"loaded image for slice{i+1}","DEVwidth_slice_stl")
        else:
            out[i].saveas(os.path.join(path,f"slice{i+1}.dwg"))

def DEVdirectional_slice(stl_path:str,output_folder:str,material_thickness:float,noturn:bool = False,printDeets:bool = False):
    tri = trimesh.load_mesh(stl_path)
    grid_pack_dwg(directional_slice_stl(tri,material_thickness,noturn,printDeets)).saveas(os.path.join(output_folder,"least_part.dwg"))

## view
def view_stl_old(stl_path: str, output_dir: str, output_name: str = "stl_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave:bool = False):
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


    plt.savefig(f"{output_dir}/{output_name}")
    if not onlySave:
        plt.show()

    if return_info:
        info = {
            'bounding_box': ((xlim[0], xlim[1]), (ylim[0], ylim[1]), (zlim[0], zlim[1])),
            'num_entities': len(target_stl.vectors),
            'filename': os.path.splitext(os.path.basename(stl_path))[0]
        }
        return plt
    return plt

def view_stl_sub(stl_path: str, output_dir: str, output_name: str = "stl_view.png", color_triangles: bool = True, keep_aspect_ratio: bool = True, rotation_angles: tuple = None, resolution: tuple = (1600, 1200), margin: float = 0.1):
    # Load the STL file
    your_mesh = m.Mesh.from_file(stl_path)
    
    # Extract vertices and faces
    vertices = your_mesh.vectors.reshape(-1, 3)
    faces = your_mesh.vectors
    
    # Rotate the vertices if rotation_angles are provided
    if rotation_angles is not None:
        vertices = rotate(vertices, rotation_angles)
        faces = vertices.reshape(-1, 3, 3)
    
    # Create a new plot
    fig = plt.figure(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Optionally color each triangle differently
    if color_triangles:
        face_colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))
    else:
        face_colors = "blue"
    
    # Create a Poly3DCollection
    collection = Poly3DCollection(faces, facecolors=face_colors, edgecolor='k')
    ax.add_collection3d(collection)
    
    # Compute the range for each axis
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    
    # Compute the margins
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    z_margin = (z_max - z_min) * margin
    
    # Adjust aspect ratio if required
    if keep_aspect_ratio:
        # Find the largest dimension to scale equally
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        # Calculate midpoints
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2

        # Set the limits to keep aspect ratio with margins
        ax.set_xlim(mid_x - max_range / 2 - x_margin, mid_x + max_range / 2 + x_margin)
        ax.set_ylim(mid_y - max_range / 2 - y_margin, mid_y + max_range / 2 + y_margin)
        ax.set_zlim(mid_z - max_range / 2 - z_margin, mid_z + max_range / 2 + z_margin)
    else:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    # Automatically adjust camera angles if rotation_angles are not provided
    if rotation_angles is None:
        ax.view_init(elev=90, azim=-90)
    else:
        ax.view_init(elev=20, azim=30)
    
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, output_name))
    plt.show()

def view_dwg(dwg_path: str, output_dir: str, output_name: str = "dwg_view.png", start_end_points: tuple = None, return_info: bool = False, onlySave: bool = False, resolution: tuple = (1600, 1200)):
    """
    Render a DWG file to an image file.

    :param dwg_path: Path to the input DWG file
    :param output_dir: Directory to save the output image
    :param output_name: Name of the output image file
    :param start_end_points: Optional tuple of start and end points to highlight
    :param return_info: If True, returns the information about the entities in the DWG file
    :param onlySave: If True, only saves the image without showing it
    :param resolution: Resolution of the output image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the DWG file
    doc = ezdxf.readfile(dwg_path)
    msp = doc.modelspace()
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(resolution[0] / 100, resolution[1] / 100), dpi=100)
    ax.set_aspect('equal')
    
    # Plot entities
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            ax.plot([start[0], end[0]], [start[1], end[1]], color='blue', lw=1)
        
        elif entity.dxftype() == 'LWPOLYLINE':
            points = list(entity.vertices())  # Get all vertices
            points.append(points[0])  # Close the polyline if it's closed
            x, y = zip(*points)  # Separate x and y coordinates
            colorRGB = tuple(np.array(ezdxf.colors.aci2rgb(entity.dxf.color))/255)
            ax.plot(x, y, color=colorRGB, lw=1)
            
    
    # Highlight start and end points if provided
    if start_end_points:
        start_point, end_point = start_end_points
        ax.plot(start_point[0], start_point[1], 'go')  # Start point
        ax.plot(end_point[0], end_point[1], 'ro')    # End point
    
    # Set axis limits
    ax.autoscale_view()
    
    # Save image
    image_path = os.path.join(output_dir, output_name)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    
    # Optionally show image
    if not onlySave:
        plt.show()
    
    # Return information if requested
    if return_info:
        info = {"lines": [], "polylines": [], "arcs": []}
        for entity in msp:
            if entity.dxftype() == 'LINE':
                info["lines"].append({
                    "start": entity.dxf.start,
                    "end": entity.dxf.end
                })
            elif entity.dxftype() == 'LWPOLYLINE':
                info["polylines"].append({
                    "points": list(entity.vertices())
                })
            elif entity.dxftype() == 'ARC':
                info["arcs"].append({
                    "center": entity.dxf.center,
                    "radius": entity.dxf.radius,
                    "start_angle": entity.dxf.start_angle,
                    "end_angle": entity.dxf.end_angle
                })
        return info
    
    print(f"DWG view saved as {image_path}")


def view_stl(stl_path:str):
    
    data = m.Mesh.from_file(stl_path)
    # reshape Nx3x3 to N*3x3 to convert polygons to points
    points = data.vectors.reshape(-1, 3)
    faces_index = np.arange(len(points))  # get indexes of points

    N=3
    rows = (np.arange(points.shape[0])%N == (N-1)) + 1
    triangles = np.repeat(points, rows, axis=0)
    triangles = np.insert(triangles, [*range(4, len(triangles), 4)], [np.nan, np.nan, np.nan], axis=0)

    lines = [
            go.Mesh3d(
            x = points[:,0],  # pass first column of points array
            y = points[:,1],  # .. second column
            z = points[:,2],  # .. third column
            # i, j and k give the vertices of triangles
            i = faces_index[0::3],  # indexes of the points. k::3 means pass every third element 
            j = faces_index[1::3],  # starting from k element
            k = faces_index[2::3],  # for example 2::3 in arange would be 2,5,8,...
            opacity = 1,
            color='lightpink'
        ),
        go.Scatter3d(
            x=triangles[:,0],  # pass first column of points array
            y=triangles[:,1],  # .. second column
            z=triangles[:,2],  # .. third column
            mode='lines',
            name='',
            line=dict(color='rgb(25,25,25)', width=7)
        )]
    
    
    fig = go.Figure(data=lines)
    fig.update_layout(width=1000, height=1000)

    fig.show()


def view_drawing(doc: ezdxf.document.Drawing, save_path: str = None):
    doc.saveas(os.path.join(dwg_output_dir,"temp.dwg"))
    view_dwg(os.path.join(dwg_output_dir,"temp.dwg"),image_output_dir,"temp.png")


def save_drawing(drawing: ezdxf.document.Drawing, output_directory:str,filename: str = "out", dxf_version: str = 'R12'):
    """
    Saves an ezdxf.document.Drawing to a file with the specified DXF version.

    Parameters:
    drawing (ezdxf.document.Drawing): The DXF document to save.
    filename (str): The filename where the drawing will be saved (e.g., 'output.dxf').
    dxf_version (str): The DXF version to save the file as. Supported versions include 'R12', 'R14', 'R2000', etc.
                       Default is 'R12'.
    
    Returns:
    None
    """
    # Set the DXF version of the document
    drawing.dxfversion = dxf_version

    # Save the document to the specified filename
    drawing.saveas(os.path.join(output_directory,filename+".dwg"))
