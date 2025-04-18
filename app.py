from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from flask_cors import CORS
import google.generativeai as genai
import json
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for API calls

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model/xray_anomaly_detector.h5")
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

def enhance_image(img):
    """
    Enhance X-ray image quality
    """
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    return denoised

def preprocess_image(image):
    """
    Preprocess the input image for model prediction
    """
    try:
        # Read and decode image
        file_bytes = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image")
        
        print(f"Original image shape: {img.shape}")
        
        # Enhance image quality
        img = enhance_image(img)
        
        # Resize to model's expected size
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values to [-1,1] (common for medical imaging)
        img = (img.astype(np.float32) - 127.5) / 127.5
        
        # Add batch and channel dimensions
        img = np.expand_dims(img, axis=[0, -1])
        print(f"Preprocessed image shape: {img.shape}")
        print(f"Value range: [{np.min(img)}, {np.max(img)}]")
        
        return img
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def get_prediction_and_confidence(model_output):
    """
    Convert model output to prediction and confidence
    """
    # Get raw prediction value
    raw_value = float(model_output)
    print(f"Raw model output: {raw_value}")
    
    # Apply sigmoid activation
    probability = float(tf.nn.sigmoid(raw_value).numpy())
    print(f"After sigmoid: {probability}")
    
    # Ensure probability is between 0 and 1
    probability = np.clip(probability, 0, 1)
    
    # Invert the prediction (assuming model might be trained with reversed labels)
    is_pneumonia = probability < 0.5
    
    # Calculate confidence (0-100%)
    if is_pneumonia:
        confidence = (1 - probability) * 100
        prediction = "Pneumonia"
    else:
        confidence = probability * 100
        prediction = "Normal"
        
    return prediction, confidence, probability

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image = request.files["file"]
        if image.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Please upload a PNG or JPEG image"}), 400

        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        print("\nMaking prediction...")
        model_output = model.predict(processed_image, verbose=1)[0][0]
        
        # Get prediction and confidence
        result, confidence, probability = get_prediction_and_confidence(model_output)
        
        # Round confidence to 2 decimal places
        confidence = round(confidence, 2)
        
        print(f"\nPrediction Results:")
        print(f"Final result: {result}")
        print(f"Confidence: {confidence}%")
        print(f"Raw probability: {probability}")

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "status": "success",
            "debug_info": {
                "raw_output": float(model_output),
                "probability": float(probability),
                "threshold": 0.5,
                "inverted_logic": True
            }
        })

    except ValueError as ve:
        print(f"ValueError in prediction: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Unexpected error in prediction: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json
        user_message = data.get("message", "")
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received message: {user_message}")
        
        # Generate response using Gemini
        response = gemini_model.generate_content(user_message)
        
        # Extract the text from the response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        print(f"Gemini response: {response_text}")
        
        return jsonify({
            "response": response_text,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in chatbot: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
