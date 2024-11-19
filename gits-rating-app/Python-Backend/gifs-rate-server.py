import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from flask_cors import CORS
import frames_extractor
from processor import process_gif

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
def process_gif(gif_path):
    rate = process_gif(gif_path)
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
