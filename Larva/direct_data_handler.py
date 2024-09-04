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
def DEVremoveUnneededlines(path_to_dwg:str,output_path:str,triangleErrorRange:float = 0,output_name:str="fixed_dwg",printDeets:bool = True):
    removeUnneededlines(ezdxf.readfile(path_to_dwg),triangleErrorRange,printDeets).saveas(os.path.join(output_path,output_name+".dwg"))

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

def get_stl_info(path_to_stl:str,output_path:str,outputfoldername:str = "stl_file_infos",printDeets:bool = True):
    output_path = os.path.join(output_path,outputfoldername)
    os.mkdir(output_path)
    body_path = os.path.join(output_path,"bodies")
    body_image_path = os.path.join(output_path,"body_images")
    os.mkdir(body_path)
    os.mkdir(body_image_path)
    out = DEVview_stl(path_to_stl,output_path,"stl.png",None,True,True)
    bounding_box = out[1]["bounding_box"]

    triangles_path = os.path.join(output_path,"triangles")
    triangles_image_path = os.path.join(output_path,"triangle_images")
    os.mkdir(triangles_image_path)

    return_stl_parts(path_to_stl,output_path,"triangles",printDeets)
    files = [i for i in os.listdir(triangles_path) if i.lower().endswith(".stl")]
    triangles= [[i+1 for i in range(len(files))],[],[],[]]
    for i in range(len(files)):
        file_path = os.path.join(triangles_path,files[i])
        target_stl = m.Mesh.from_file(file_path)
        triangles[1].append(str(target_stl.vectors[0][0]))
        triangles[2].append(str(target_stl.vectors[0][1]))
        triangles[3].append(str(target_stl.vectors[0][2]))
        DEVview_stl(file_path,triangles_image_path,f"triangle{i+1}.png",bounding_box,False,True)
        printIF(printDeets,f"{i+1}/{len(files)} triangles loaded","get_stl_info")

    png_to_mp4(triangles_image_path,output_path,"triangles")

    createSimpleXLSX(["#","p1","p2","p3"],triangles,output_path,"info")

    DEVreturn_stl_bodies(path_to_stl,output_path,"bodies",printDeets)
    files = [i for i in os.listdir(body_path) if i.lower().endswith(".stl")]
    for i in range(len(files)):
        file_path = os.path.join(body_path,files[i])
        view_stl(file_path,body_image_path,f"Body_{i}.png",bounding_box,False,True)
        printIF(printDeets,f"Created image for body {i+1}/{len(files)}","get_stl_info")
    png_to_mp4(body_image_path,output_path,"bodies")

def DEVreturn_dwg_parts(path_to_dwg: str, outputdir: str, foldername: str = "dwgparts",printDeets:bool = True):
    out = os.path.join(outputdir,foldername)
    os.mkdir(out)
    data = return_dwg_parts(ezdxf.readfile(path_to_dwg),printDeets)
    for i in range(len(data)):
        data[i].saveas(os.path.join(out,f"part{i+1}.dwg"))
    
def DEVreturn_stl_parts(path_to_stl: str, outputdir: str, foldername: str = "stlparts",printDeets:bool = True):
    out = os.path.join(outputdir,foldername)
    os.mkdir(out)
    data = return_dwg_parts(m.Mesh.from_file(path_to_stl),printDeets)
    for i in range(len(data)):
        data[i].save(out,f"part{i+1}.stl")

def DEVreturn_stl_bodies(path_to_stl: str, outputdir: str, foldername: str = "stlbodies",printDeets:bool = True):
    out = os.path.join(outputdir,foldername)
    os.mkdir(out)
    data = return_stl_bodies(m.Mesh.from_file(path_to_stl),printDeets)
    for i in range(len(data)):
        data[i].save(out,f"body{i+1}.stl")

def DEVslice_stl_to_dwg(stl_path: str, slicing_plane_normal: list, slicing_plane_point:list, output_dir: str, output_name: str = "sliced",printDeets:bool = True):
    slice_stl_to_dwg(m.Mesh.from_file(stl_path),slicing_plane_normal,slicing_plane_point,printDeets).saveas(os.path.join(output_dir,output_name+".dwg"))

def DEVwidth_slice_stl(stl_path:str,outputFolder:str,sliceWidth:float,slicePlaneNormal:list = [0,0,1],outputFolderName:str = "stl_slices",printDeets:bool = False,withExtra:bool = True):
    out = width_slice_stl(m.Mesh.from_file(stl_path),sliceWidth,slicePlaneNormal,printDeets)
    path = os.path.join(outputFolder,outputFolderName)
    os.mkdir(path)
    if withExtra:
        os.mkdir(os.path.join(path,"image"))
        os.mkdir(os.path.join(path,"data"))
    for i in range(len(out)):
        if withExtra:
            out[i].saveas(os.path.join(path,os.path.join("data",f"slice{i+1}.dwg")))
            view_dwg(os.path.join(path,os.path.join("data",f"slice{i+1}.dwg")),os.path.join(path,"image"),f"slice{i+1}.png")
            printIF(printDeets,f"loaded image for slice{i+1}","DEVwidth_slice_stl")
        else:
            out[i].saveas(os.path.join(path,f"slice{i+1}.dwg"))

def stl_to_gltf(stl_path: str, outputpath: str, outputname: str = "out",printDeets:bool = True):
    mesh = tri.load_mesh(stl_path)
    
    if not mesh.is_volume:
        raise ValueError("The mesh is not a valid volume.")

    gltf_path = f"{outputpath}/{outputname}.gltf"
    
    mesh.export(gltf_path)
    
    printIF(printDeets,f"STL file '{stl_path}' has been successfully converted to GLTF '{gltf_path}'","stl_to_gltf")
