import main as dh

dh.clear_output_paths()
dh.slice_obj_file("data/obj/teddy_bear.obj",5,dh.dwg_output_dir)
dh.view_obj("data/obj/teddy_bear.obj",dh.image_output_dir)
dh.view_dwg("output/CAD/dwg/slice_0.dwg",dh.image_output_dir)
dh.slice_stl_to_dwg("data/stl/plane.stl",[("x",0)],dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/sliced_stl_1.dwg",dh.image_output_dir,"stltodwg.png")
dh.view_stl("data/stl/plane.stl",dh.image_output_dir)