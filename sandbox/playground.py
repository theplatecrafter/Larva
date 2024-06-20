import Larva.data_handler as dh

#complex_stl/thankyou_for_listening.stl
path = "data/stl/complex_stl/thankyou_for_listening.stl"

dh.clear_output_paths()
dh.view_stl(path,dh.image_output_dir)
dh.slice_stl_to_dwg(path,[0,0,1],[0,0,0],dh.dwg_output_dir,"sliced")
dh.view_dwg("output/CAD/dwg/sliced.dwg",dh.image_output_dir,"before",None,False,False,(6500,600))
dh.removeUnneededlinesFromFile("output/CAD/dwg/sliced.dwg",dh.dwg_output_dir)
dh.view_dwg("output/CAD/dwg/fixed_dwg.dwg",dh.image_output_dir,"after",None,False,False,(6500,600))