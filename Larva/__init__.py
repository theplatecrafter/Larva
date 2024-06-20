import os

# set output folders
body_output_dir = "output/CAD/bodies"
dwg_output_dir = "output/CAD/dwg"
image_output_dir = "output/image"
obj_output_dir = "output/CAD/obj"
stl_output_dir = "output/CAD/stl"
info_output_dir = "output/infos"
video_output_dir = "output/video"
slices_output_dir = "output/slices"

try:
    os.mkdir("output")
    os.mkdir("output/CAD")
    os.mkdir(body_output_dir)
    os.mkdir(dwg_output_dir)
    os.mkdir(image_output_dir)
    os.mkdir(obj_output_dir)
    os.mkdir(stl_output_dir)
    os.mkdir(info_output_dir)
    os.mkdir(video_output_dir)
    os.mkdir(slices_output_dir)
except Exception:
    pass
