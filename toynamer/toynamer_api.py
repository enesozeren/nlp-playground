from flask import Flask, request, jsonify
import torch
from utils import generate_name

app = Flask(__name__)

# Load your trained PyTorch model
toynamer = torch.load('toynamer/rnn_turkish_name_generator.pth', map_location=torch.device('cpu'))
toynamer.eval()

@app.route('/generate', methods=['POST'])
def generate():
    # Check if 'temperature' parameter exists in the request
    if 'temperature' not in request.json:
        return jsonify({'error': 'No temperature parameter provided'})

    # Get the temperature parameter from the request
    temperature = request.json['temperature']

    # Convert the temperature parameter to float
    try:
        temperature = float(temperature)
    except ValueError:
        return jsonify({'error': 'Invalid temperature parameter'})

    # Perform prediction using the model
    output = generate_name(toynamer, temperature)

    # Return the prediction
    return jsonify({'name': output})

if __name__ == '__main__':
    app.run(debug=True)