import main as dh

dh.clear_output_paths()

dh.slice_stl_to_dwg_without_lines("data/plane.stl",[("x",0)],dh.dwg_output_dir) ## new slicer
dh.slice_stl_to_dwg("data/plane.stl",[("x",0)],dh.dwg_output_dir,"sliced_stl_old") ## old slicer

dh.view_dwg("output/CAD/dwg/sliced_stl_1.dwg",dh.image_output_dir)
dh.view_dwg("output/CAD/dwg/sliced_stl_old1.dwg",dh.image_output_dir,"dwg_view_old")