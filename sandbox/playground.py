import Larva.direct_data_handler as dh

#complex_stl/thankyou_for_listening.stl
path = "data/stl/cube.stl"

dh.clear_output_paths()
dh.startreadingterminal()

dh.view_stl_dev(path,dh.image_output_dir)

dh.stopreadingterminal()