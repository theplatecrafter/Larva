from Larva import direct_data_handler as dh
path = "data/dwg/autocad_square2.dxf"

z = dh.ezdxf.readfile(path)
print(z.dxfversion)
dh.view_drawing(z,dh.os.path.join(dh.dwg_output_dir,"out.dxf"))