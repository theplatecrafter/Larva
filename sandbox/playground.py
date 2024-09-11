import Larva.direct_data_handler as dh

path = "data/stl/hollow_cube.stl"

dh.clear_output_paths()
#dh.startreadingterminal()

dh.smart_slice_stl(dh.trimesh.load_mesh(path),printDeets=False)

#dh.stopreadingterminal()