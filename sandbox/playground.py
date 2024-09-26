import Larva.direct_data_handler as dh

path = "data/stl/complex_stl/Stanford_Bunny_sample.stl"
tri = dh.trimesh.load_mesh(path)

dh.clear_output_paths()


##dh.startreadingterminal()

dh.Bs_view_stl(path)

exit()

d = dh.grid_pack_dwg(dh.width_slice_stl(tri,4),4,2,False)
dh.view_drawing(d,dh.image_output_dir)

exit()

d = dh.smart_slice_stl(tri,4,printDeets=False)
dh.view_drawing(d,dh.image_output_dir)

##dh.stopreadingterminal()

