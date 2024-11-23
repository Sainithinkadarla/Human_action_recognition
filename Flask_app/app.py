from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

# Load pre-trained model
mobilenetv3large = load_model("static/Models/mobilenetv3large.keras")

train_data = pd.read_csv("Training_set.csv")
label_mapping = dict(enumerate(train_data['label'].unique()))

# Video processing function
def process_video_with_action_sequence(video_path, model, label_mapping, frame_skip=30):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    
    frame_predictions = []
    frame_confidences = []
    frame_numbers = []
    
    frame_number = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            frame_resized = cv2.resize(frame, (160, 160))
            frame_array = np.expand_dims(frame_resized, axis=0)

            predictions = model.predict(frame_array, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            frame_predictions.append(predicted_class)
            frame_confidences.append(confidence)
            frame_numbers.append(frame_number)

        frame_number += 1

    video.release()

    frame_predictions_named = [label_mapping[pred] for pred in frame_predictions]

    action_sequence = []
    for i in range(len(frame_predictions_named)):
        if i == 0 or frame_predictions_named[i] != frame_predictions_named[i - 1]:
            action_sequence.append(frame_predictions_named[i])

    action_confidences = {
        action: np.mean([frame_confidences[j] for j in range(len(frame_predictions_named)) if frame_predictions_named[j] == action])
        for action in set(frame_predictions_named)
    }

    return {
        "total_frames": frame_count,
        "frames_processed": len(frame_predictions),
        "action_sequence": action_sequence,
        "action_confidences": action_confidences,
        "frame_predictions": frame_predictions_named,
        "frame_confidences": frame_confidences,
        "frame_numbers": frame_numbers,
    }


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join('uploads', filename)
    video.save(video_path)

    # Process video
    results = process_video_with_action_sequence(video_path, mobilenetv3large, label_mapping)

    # Convert NumPy types to Python types for JSON serialization
    results = {
        key: (
            [float(x) if isinstance(x, np.float32) else x for x in value]
            if isinstance(value, list) else
            {k: float(v) if isinstance(v, np.float32) else v for k, v in value.items()}
            if isinstance(value, dict) else
            float(value) if isinstance(value, np.float32) else value
        )
        for key, value in results.items()
    }

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
