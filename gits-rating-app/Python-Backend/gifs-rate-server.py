import os
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from processor import process_gif

app = Flask(__name__, static_folder='static/react')
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

num_of_frames = 5
height = 128
width = 128

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route('/rate-gif', methods=['POST'])
def upload_file():
    # Check if a file is part of the request
    if 'gif' not in request.files:
        return jsonify(error='No file uploaded.'), 400

    file = request.files['gif']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        rate = process_gif(file_path)
        if rate == 0:
            return jsonify(rating='Appropriate')
        else:
            return jsonify(rating='Inappropriate')

    return jsonify(error='Only GIF files are allowed!'), 400

if __name__ == '__main__':
    app.run(port=5000)
