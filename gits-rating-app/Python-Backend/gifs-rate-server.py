import random

import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from flask_cors import CORS

import frames_extractor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

num_of_frames = 5
height = 128
width = 128

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to process the GIF and return a rating
def process_gif(file_path):
    frames = frames_extractor.extract_frames(file_path, width, height, num_of_frames)
    frames = np.array(frames).reshape(-1, num_of_frames, height, width, 3)
    model = tf.keras.models.load_model('../../convlstm_model___Date_Time_2024_10_02__12_28_57___Loss_0.9139771461486816___Accuracy_0.6860841512680054.h5')
    predicted = model.predict(frames)
    print(predicted)
    rate = list(predicted[0]).index(max(predicted[0])) + 1
    if rate == 0:
        return 'Appropriate'
    else:
        return 'Inappropriate'


@app.route('/rate-gif', methods=['POST'])
def upload_file():
    # Check if a file is part of the request
    if 'gif' not in request.files:
        return jsonify(error='No file uploaded.'), 400

    file = request.files['gif']

    # If the file is not a valid GIF
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the GIF and get the rating
        rating = process_gif(file_path)
        return jsonify(rating=rating)

    return jsonify(error='Only GIF files are allowed!'), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)
