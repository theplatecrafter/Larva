import Larva.direct_data_handler as dh

#complex_stl/thankyou_for_listening.stl
path = "data/stl/complex_stl/rhombic.stl"

dh.clear_output_paths()
#dh.startreadingterminal()

dh.DEVslice_stl_to_dwg(path,[0,0,1],[0,0,0],dh.dwg_output_dir,printDeets=True)
dh.view_dwg("output/CAD/dwg/sliced.dwg",dh.image_output_dir)
dh.DEVview_stl(path,dh.image_output_dir)

#dh.stopreadingterminal()