import main as dh
from tools import *

path = "data/stl/complex_hollow_cube.stl"

dh.clear_output_paths()

dh.view_stl(path,dh.image_output_dir)
dh.slice_stl_to_dwg(path,[("x",0)],dh.dwg_output_dir)
dh.get_dwg_info("output/CAD/dwg/sliced_stl_1.dwg","output")

dh.removeUnneededlines("output/CAD/dwg/sliced_stl_1.dwg",dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/fixed_dwg.dwg",dh.image_output_dir,"fixed.png")

