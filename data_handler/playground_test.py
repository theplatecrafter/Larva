import main as dh
from tools import *

path = "data/stl/complex_hollow_cube.stl"

dh.clear_output_paths()

dh.get_dwg_info("data/dwg/hollow_cube.dwg",dh.info_output_dir)