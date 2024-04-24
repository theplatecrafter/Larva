import data_handler as dh

dh.view_dwg(f"{dh.dwg_output_dir}/sliced_stl.dwg",dh.image_output_dir)

dh.view_stl("data/plane.stl",dh.image_output_dir)

dh.slice_stl_to_dwg("data/plane/stl",[("x",0)],dh.dwg_output_dir)

dh.extract_bodies_from_stl("data/plane.stl",dh.body_output_dir)