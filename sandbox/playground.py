import Larva.direct_data_handler as dh

path = "data/stl/complex_stl/Play_Brick_3mm.STL"
tri = dh.trimesh.load_mesh(path)

dh.clear_output_paths()

dh.Bs_view_stl(path)

exit()

out = dh.directional_slice_stl(tri,4,printDeets=True)
d = dh.grid_pack_dwg(out)


dh.view_drawing(d,dh.image_output_dir)
