from flask import Flask, request, jsonify
import numpy as np
from pyngrok import ngrok
import os
import io
import tensorflow as tf
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
CORS(app, origins="*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file with the key 'file'
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is a PNG image
    if file.mimetype != 'image/png':
        return jsonify({'error': 'Unsupported file format. Only PNG images are supported.'})

    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(file.read()), target_size=(28, 28), color_mode='grayscale')
    input_data = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Make prediction
    result = model.predict(input_data)
    predicted_class = np.argmax(result, axis=1)[0]

    return jsonify({'prediction': int(predicted_class)})

@app.route('/', methods=['GET'])
def hello():
    return 'Hello from server!'

if __name__ == '__main__':
 # Get the port from the environment variable, or use 5000 if not set
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)