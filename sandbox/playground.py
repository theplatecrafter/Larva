import Larva.direct_data_handler as dh

path = "data/stl/complex_stl/thankyou_for_listening.stl"
tri = dh.trimesh.load_mesh(path)

dh.clear_output_paths()
##dh.startreadingterminal()

dh.Bs_view_stl(path)
exit()

d = dh.smart_slice_stl(tri,1,printDeets=False)
dh.view_drawing(d,dh.image_output_dir)

##dh.stopreadingterminal()