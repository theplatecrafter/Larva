import Larva.direct_data_handler as dh

path = "data/stl/test_subject.stl"
tri = dh.resize_mesh(dh.trimesh.load_mesh(path),1)

dh.clear_output_paths()


dh.view_drawing(dh.slice_stl(tri,[0,0,1],[0,0,5]))


out = dh.gided_layer_slice_stl(tri,4,[0,0,1])
d = dh.grid_pack_dwg(out)


d.saveas(dh.os.path.join(dh.dwg_output_dir,"out.dwg"))

dh.view_dwg(dh.os.path.join(dh.dwg_output_dir,"out.dwg"),dh.image_output_dir)

exit()

## TODO: maybe change lwpolyline to polyline? for corel draw, or maybe remove the combine_lines thing so that all the lines are seperate