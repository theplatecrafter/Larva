import Larva as dh

path = "data/stl/complex_hollow_cube.stl"

dh.clear_output_paths()
dh.view_stl(path,dh.image_output_dir)
dh.width_slice_stl(path,dh.slices_output_dir,1.0,[0,0,1],"stl_slices",True,True)
