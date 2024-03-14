from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the InceptionV3 model
model = load_model('InceptionV3_1.h5')

# Define a function to preprocess the image
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))  # InceptionV3 input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # print("Shape of processed image:", img_array.shape)
        return img_array
    except Exception as e:
        print("Error preprocessing image:", str(e))
        return None

# Define a route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    response = app.make_response(jsonify({'message': 'success'}))
    response.headers['Access-Control-Allow-Origin'] = '*'  # Set the appropriate origin
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    if request.method == 'POST':
        
        print("image: ", request.files['file'])
        
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Check if the file format is supported
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Preprocess the image
            img_path = 'temp_image.jpg'  # Temporary image file
            file.save(img_path)
            
            # print("file saved: ", file)
            test_image_path = img_path
            img = image.load_img(test_image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            processed_image = preprocess_input(img_array)

            # Make prediction
            predictions = model.predict(processed_image)
            # print(predictions)
            # Get predicted class
            predicted_class = np.argmax(predictions)
            print(predicted_class)

            # Print predicted class
            print("Predicted class:", predicted_class)
            return jsonify({'class_id': int(predicted_class)})
        else:
            return jsonify({'error': 'Unsupported file format'})
        
# @app.route('/report', methods=['POST'])
# def report():
#     if request.method == 'POST':
        
#         print("image: ", request.files['file'])
        
#         # Check if an image file is uploaded
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})
        
#         file = request.files['file']
        
#         # Check if the file is empty
#         if file.filename == '':
            
#             return jsonify({'error': 'No selected file'})
        
#         # Check if the file format is supported
#         if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Preprocess the image
#             unique_filename = datetime.now().strftime("%Y%m%d_%H%M%S%f") + os.path.splitext(file.filename)[1]
#             img_path = os.path.join('Report', unique_filename)  # Unique image file path
            
#             # Save the image with the unique filename
#             file.save(img_path)
            
#             return jsonify({'message': 'File uploaded successfully', 'filename': unique_filename})
#         else:
#             return jsonify({'error': 'Unsupported file format'})
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


