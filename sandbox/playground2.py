import Larva as dh

path = "data/stl/cube.stl"

dh.clear_output_paths()
dh.width_slice_stl(path,dh.slices_output_dir,1.0,[0,0,1],"stl_slices",True)
dh.get_dwg_info("output/slices/stl_slices/slicesDATA/slice1.dwg",dh.info_output_dir)