import Larva.direct_data_handler as dh

path = "data/stl/test_subject.stl"
tri = dh.trimesh.load_mesh(path)

dh.clear_output_paths()
##dh.startreadingterminal()

dh.Bs_view_stl(path)

d = dh.grid_pack_dwg(dh.width_slice_stl(tri,4),4,3,True)
dh.view_drawing(d,dh.image_output_dir)

##dh.stopreadingterminal()

exit()

d = dh.smart_slice_stl(tri,4,printDeets=False)
dh.view_drawing(d,dh.image_output_dir)