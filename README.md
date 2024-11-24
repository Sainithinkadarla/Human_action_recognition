# Human Action Recognition Using Deep Learning Models

This project implements a comparative study of different Convolutional Neural Network (CNN) architectures for human action recognition using video data. The project also includes a Flask-based web application to process uploaded videos and identify action sequences.

---

## Project Overview

### Features:

1. **Model Comparison**:
    - **EfficientNetB7**
    - **MobileNetV3Small**
    - **MobileNetV3Large**
2. **Training and Evaluation**:
    - Models are trained on labeled datasets.
    - Comparative metrics (accuracy and loss) are generated and visualized.
3. **Video Processing and Action Recognition**:
    - Predicts action sequences and confidence levels for uploaded videos.
    - Flask app for user-friendly interaction.
   
---

## Installation and Usage

### Prerequisites:

- Python 3.7+
- TensorFlow/Keras
- Flask
- OpenCV
- NumPy, Pandas, Matplotlib, and Seaborn

### Installation:

1. Clone the repository:
    
    ```bash

    git clone https://github.com/Sainithinkadarla/Human_action_recognition.git
    
    ```
    
2. Install dependencies:
    
    ```bash
    cd Human_action_recognition
    pip install -r requirements.txt
    
    ```
    

### Training:

Run `human-action mobilenetv3large.ipynb` for training the  model

### Running Flask App:

1. Start the server:
    
    ```bash

    python app.py
    
    ```
    
2. Open your browser and go to:
    
    ```arduino
    http://localhost:5000
    
    ```
    

---

## Usage Instructions

1. Upload a video on the Flask app interface.
2. Processed results include:
    - Detected action sequence.
    - Confidence for each detected action.
3. Visualize predictions and sequences in a structured format.
