import main as dh
from tools import *

path = "data/stl/cube.stl"

dh.clear_output_paths()
dh.view_stl(path,dh.image_output_dir)
dh.slice_stl_to_dwg(path,[0,0,1],[0,0,30],dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/sliced.dwg",dh.image_output_dir)