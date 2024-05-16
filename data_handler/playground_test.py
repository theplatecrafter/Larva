import main as dh
from tools import *

dh.clear_output_paths()
dh.view_stl("data/stl/hollow_cube.stl",dh.image_output_dir)
dh.slice_stl_to_dwg("data/stl/hollow_cube.stl",[("x",0)],dh.dwg_output_dir,"cubeSliced")
dh.get_dwg_info("output/CAD/dwg/cubeSliced1.dwg","output")
dh.view_dwg("output/CAD/dwg/cubeSliced1.dwg",dh.image_output_dir)