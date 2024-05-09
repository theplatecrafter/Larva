from tools import *
import main as dh



def tmp(dwg):
    dwgFile = ezdxf.readfile(dwg)
    msp = dwgFile.modelspace()

    lines = []
    for entity in msp:
        points = entity.get_points()
        for i in range(len(points)):
            for j in range(len(points)):
                lines.append([(points[i][0],points[i][1]),(points[j][0],points[j][1])])
    
    newLines = []
    for i in range(len(lines)):
        if lines[i][0] != lines[i][1]:
            newLines.append(lines[i])
    lines = rmSame(newLines)

    newLines = []
    for i in range(len(lines)):
        checkFor = [lines[i][1],lines[i][0]]
        if checkFor not in newLines:
            newLines.append(lines[i])
    lines = newLines

    delete = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if (intersect(lines[i],lines[j])) and not (lines[i][0] == lines[j][0] or lines[i][1] == lines[j][1] or lines[i][0] == lines[j][1] or lines[i][1] == lines[j][0]):
                delete.append(i)
                delete.append(j)
    
    delete = rmSame(delete)
    for i in range(len(delete)):
        lines.pop(delete[i]-i)

    return lines


lines = tmp("output/CAD/dwg/sliced_stl_1.dwg")
print(lines)
create_dwg_outof_lines(lines,"output/CAD/dwg/lines.dwg")
dh.view_dwg("output/CAD/dwg/lines.dwg",dh.image_output_dir)