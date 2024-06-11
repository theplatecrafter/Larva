from tools import *
import main as dh

dh.clear_output_paths()
dh.view_stl("data/stl/hollow_cube.stl",dh.image_output_dir)
print(dh.view_dwg("data/dwg/cube.dwg",dh.image_output_dir,"yep.png",None,True))