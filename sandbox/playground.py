import Larva as dh

path = "data/stl/complex_hollow_cube.stl"

dh.clear_output_paths()
dh.view_stl(path,dh.image_output_dir)
dh.slice_stl_to_dwg(path,[0,0,1],[0,0,10],dh.dwg_output_dir)
dh.removeUnneededlines("output/CAD/dwg/sliced.dwg",dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/fixed_dwg.dwg",dh.image_output_dir)