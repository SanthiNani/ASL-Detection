from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('asl_detection_model.h5')

# Define the class labels (A-Z, SPACE, DELETE, NOTHING)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE', 'NOTHING'
]

# Define the upload folder inside the static folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file to the static/uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Debug: Print the file path
        print("File saved at:", file_path)
        
        # Debug: Check if the file exists
        if os.path.exists(file_path):
            print("File exists:", file_path)
        else:
            print("File does NOT exist:", file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(64, 64))  # Resize to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        
        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        
        # Pass the relative path to the template (use forward slashes)
        relative_path = os.path.join('uploads', file.filename).replace("\\", "/")
        
        # Debug: Print the relative path
        print("Relative path for template:", relative_path)
        
        return render_template('index.html', prediction=predicted_label, image_path=relative_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)