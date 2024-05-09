import main as dh

dh.clear_output_paths()
dh.slice_stl_to_dwg("data/stl/Another_Hollow_Cube.stl",[("x",0)],dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/sliced_stl_1.dwg",dh.image_output_dir,)
dh.view_stl("data/stl/Another_Hollow_Cube.stl",dh.image_output_dir)