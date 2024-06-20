from Larva import direct_data_handler as dfm
path = "data/stl/complex_stl/thankyou_for_listening.stl"

dfm.clear_output_paths
dfm.view_stl(path,dfm.image_output_dir,"stl.png",True,True,None,(20000,15000))