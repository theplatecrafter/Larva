import Larva.direct_data_handler as dh

path = "data/stl/test_subject.stl"
tri = dh.trimesh.load_mesh(path)

dh.clear_output_paths()


#dh.Bs_view_stl(path)


d = dh.grid_pack_dwg(dh.directional_slice_stl(tri,4,printDeets=False))
dh.view_drawing(d,dh.image_output_dir)
