import Larva.direct_data_handler as dh

#complex_stl/thankyou_for_listening.stl
path = "data/stl/cube.stl"

dh.clear_output_paths()
dh.startreadingterminal()

dh.DEVwidth_slice_stl(path,dh.slices_output_dir,1,[0,0,1],printDeets=True)

dh.stopreadingterminal()