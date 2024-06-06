import main as dh
from tools import *

path = "data/stl/cube.stl"

dh.clear_output_paths()

dh.view_stl(path,dh.image_output_dir)
dh.slice_stl_to_dwg(path,[("x",0)],dh.dwg_output_dir)


dh.get_dwg_info("output/CAD/dwg/sliced_stl_1.dwg","output/infos","original_info")

dh.removeUnneededlines("output/CAD/dwg/sliced_stl_1.dwg",dh.dwg_output_dir)

dh.get_dwg_info("output/CAD/dwg/fixed_dwg.dwg","output/infos","fixed_info")
