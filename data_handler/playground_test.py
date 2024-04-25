import main as dh

dh.slice_stl_to_dwg("data/Cube_3d_printing_sample.stl",[("x",0),("x",0),("x",0),("x",0),("x",0),("x",0)],dh.dwg_output_dir)

dh.view_dwg("output/CAD/dwg/sliced_stl_1.dwg",dh.image_output_dir)