---

# American Sign Language (ASL) Detection

![ASL Detection](http://127.0.0.1:5000)  


## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Training](#model-training)
8. [Web Application](#web-application)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
This project is an **American Sign Language (ASL) Detection System** that uses a Convolutional Neural Network (CNN) to classify ASL alphabet signs from images. The system can detect 29 classes, including the letters A-Z and additional symbols like SPACE, DELETE, and NOTHING. A Flask-based web application allows users to upload ASL images and get real-time predictions.

---

## Features
- **ASL Alphabet Detection**: Detects 26 ASL alphabet signs (A-Z) and 3 additional symbols (SPACE, DELETE, NOTHING).
- **Web Interface**: A user-friendly web interface for uploading images and viewing predictions.
- **Real-Time Prediction**: Predicts the ASL sign in real-time using a pre-trained CNN model.
- **Dataset**: Uses a publicly available ASL dataset for training and testing.

---

## Technologies Used
- **Python**: Primary programming language.
- **TensorFlow/Keras**: For building and training the CNN model.
- **Flask**: For creating the web application.
- **Bootstrap**: For styling the web interface.
- **NumPy**: For numerical computations.
- **Pillow**: For image preprocessing.
- **Scikit-learn**: For evaluation metrics (optional).

---

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/asl-detection.git
   cd asl-detection
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Download the ASL dataset from [this link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
   - Place the dataset in the `data` folder.

4. **Train the Model**:
   If you want to train the model from scratch, run:
   ```bash
   python combined_app.py
   ```
   If you run this file the model will be save as **asl_detection_model.h5** then run the app.py file in your command prompt

5. **Run the Flask App**:
   Start the Flask web application:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Usage
1. **Upload an Image**:
   - Click the "Choose File" button to upload an ASL image.
   - Click "Upload and Predict" to get the prediction.

2. **View the Prediction**:
   - The predicted ASL sign (e.g., "A", "B", "SPACE") will be displayed below the upload button.
   - The uploaded image will also be displayed.

---

## Output of the Project
![Screenshot 2025-03-04 112712](https://github.com/user-attachments/assets/13ccc05d-1309-4c93-997a-b8039566dc8c)
![Screenshot 2025-03-04 112820](https://github.com/user-attachments/assets/3009f54c-56a4-4e6c-88e9-e2573d5544b9)



## Dataset
The dataset used for this project contains 87,000 images of ASL signs, divided into 29 classes (26 letters + 3 symbols). Each class has separate folders for training and testing.

- **Dataset Link**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Dataset Structure**:
  ```
  asl_alphabet_train/
      A/
      B/
      ...
      Z/
      SPACE/
      DELETE/
      NOTHING/
  ```

---

## Model Training
The CNN model is trained using TensorFlow/Keras. The model architecture consists of:
- **Convolutional Layers**: For feature extraction.
- **Max Pooling Layers**: For downsampling.
- **Dense Layers**: For classification.
- **Dropout**: To prevent overfitting.

### Training Script
To train the model, run:
```bash
python combined_app.py
```
The trained model will be saved as `asl_detection_model.h5`.

---

## Web Application
The Flask web application provides a simple interface for users to upload ASL images and get predictions. The app uses the pre-trained model to classify the uploaded images.

### Running the Web App
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.

---

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Dataset**: Thanks to the creators of the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle.
- **TensorFlow/Keras**: For providing the tools to build and train the CNN model.
- **Flask**: For enabling the creation of the web application.

---

## Contact
For questions or feedback, feel free to reach out:
- **Name**: Peddapati Santhi Raju
- **Email**: santhinani364@gmail.com
---
