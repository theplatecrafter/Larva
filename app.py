from flask import Flask, request, jsonify
import os
import trimesh
import Larva.direct_data_handler as ddh

app = Flask(__name__)

@app.route('/slice-stl', methods=['POST'])
def slice_stl():
    stlFile = request.files['stlFile']
    normalX = float(request.form['normalX'])
    normalY = float(request.form['normalY'])
    normalZ = float(request.form['normalZ'])
    pointX = float(request.form['pointX'])
    pointY = float(request.form['pointY'])
    pointZ = float(request.form['pointZ'])

    # Save uploaded file to temp directory
    file_location = f"temp/{stlFile.filename}"
    os.makedirs('temp', exist_ok=True)
    stlFile.save(file_location)

    # Prepare slicing parameters
    slicing_plane_normal = [normalX, normalY, normalZ]
    slicing_plane_point = [pointX, pointY, pointZ]

    # Define output directory and name
    output_dir = "output"
    output_name = "sliced_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Call your slicing function
    ddh.DEVslice_stl(file_location, slicing_plane_normal, slicing_plane_point, output_dir, output_name)

    return jsonify({"message": f"File sliced and saved as {output_name}.dwg"})

if __name__ == '__main__':
    app.run(debug=True)
