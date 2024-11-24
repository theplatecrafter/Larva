from flask import Flask, request, jsonify
import Larva.data_handler as dh
import Larva.direct_data_handler as ddh

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    input_data = request.json['input']
    result = ddh.ping() + input_data
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
