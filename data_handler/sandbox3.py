import main as dh
from tools import *

def slice_stl_file(file_path, output_dir, direction, n_slices, outputfoldername="multisliced"):
    original_mesh = m.Mesh.from_file(file_path)
    
    # Determine the slicing axis
    if direction == 'x':
        axis = 0
        proj_axes = (1, 2)
    elif direction == 'y':
        axis = 1
        proj_axes = (0, 2)
    elif direction == 'z':
        axis = 2
        proj_axes = (0, 1)
    else:
        raise ValueError("Invalid direction. Choose 'x', 'y', or 'z'.")
    
    # Calculate the slicing intervals
    min_val = np.min(original_mesh.vectors[:, :, axis])
    max_val = np.max(original_mesh.vectors[:, :, axis])
    slice_interval = (max_val - min_val) / n_slices
    
    # Create output directory if it does not exist
    output_dir = os.path.join(output_dir, outputfoldername)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "data")):
        os.makedirs(os.path.join(output_dir, "data"))
    if not os.path.exists(os.path.join(output_dir, "image")):
        os.makedirs(os.path.join(output_dir, "image"))
    
    # Create slices
    for i in range(n_slices):
        lower_bound = min_val + i * slice_interval
        upper_bound = lower_bound + slice_interval
        
        # Create a new DWG file for the slice
        dwg = ezdxf.new()
        msp = dwg.modelspace()
        
        for triangle in original_mesh.vectors:
            if np.any((triangle[:, axis] >= lower_bound) & (triangle[:, axis] < upper_bound)):
                # Adjust the triangle to fit within the slice bounds
                sliced_triangle = np.clip(triangle[:, proj_axes], lower_bound, upper_bound)
                msp.add_lwpolyline(sliced_triangle.tolist() + [sliced_triangle[0].tolist()])
        
        # Save the DWG file
        slice_path = os.path.join(os.path.join(output_dir, "data"), f'slice_{i}.dwg')
        dwg.saveas(slice_path)
        print(f"Saved slice {i} to {slice_path}")
    
    # Generate images for the slices
    files = sorted(os.listdir(os.path.join(output_dir, "data")), key=lambda x: int(x.split('_')[1].split('.')[0]))
    for i in range(len(files)):
        file_path = os.path.join(os.path.join(output_dir, "data"), files[i])
        ##if os.path.isfile(file_path):
            ##dh.view_dwg(file_path, os.path.join(output_dir, "image"), files[i] + ".png")

dh.force_remove_all(dh.stl_output_dir)
slice_stl_file("data/stl/cube.stl","output/CAD/stl", direction='z', n_slices=10)
