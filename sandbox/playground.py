import Larva.direct_data_handler as dh

path = "data/stl/test_subject.stl"

dh.clear_output_paths()
#dh.startreadingterminal()

dh.DEVview_stl(path,dh.image_output_dir)
##dh.view_drawing(dh.slice_stl(dh.trimesh.load_mesh(path),[0,0,1],[0,0,0],printDeets=False),dh.image_output_dir)
dh.view_drawing(dh.smart_slice_stl(dh.trimesh.load_mesh(path),printDeets=False),dh.image_output_dir)

#dh.stopreadingterminal()