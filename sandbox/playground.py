import Larva.direct_data_handler as dh

path = "data/stl/test_subject.stl"
tri = dh.resize_mesh(dh.trimesh.load_mesh(path),1)

dh.clear_output_paths()


out = dh.gided_layer_slice_stl(tri,4,[0,0,1])
d = dh.grid_pack_dwg(out)
print(d.dxfversion)
d.saveas(dh.os.path.join(dh.dwg_output_dir,"out.dwg"))

dh.view_dwg(dh.os.path.join(dh.dwg_output_dir,"out.dwg"),dh.image_output_dir)

exit()

## TODO: maybe change lwpolyline to polyline? for corel draw