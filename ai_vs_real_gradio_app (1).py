
# Install Gradio
# --------------
# !pip install gradio

# Load Your Model
# ---------------

from tensorflow import keras

# Load trained model (update path as needed)
model = keras.models.load_model("/Users/yashtaneja/Desktop/Yash_Thesis/coding and model/AI_vs_Real_Model.h5")

# Define labels
class_names = ["Real", "AI-Generated"]

# Define Preprocessing + Prediction Function
# ------------------------------------------

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

def predict_image(img):
    # Ensure RGB (drop alpha channel if present)
    img = img.convert("RGB")

    # Resize to match model input
    img = img.resize((224, 224))

    # Convert to numpy
    img_array = np.array(img)

    # Expand dims for batch
    img_array = np.expand_dims(img_array, axis=0)

    # Apply same preprocessing as training
    img_array = preprocess_input(img_array)

    # Prediction
    pred = model.predict(img_array)

    if pred.shape[1] == 1:  # sigmoid output
        predicted_class = int(pred[0][0] > 0.5)
        confidence = pred[0][0] if predicted_class == 1 else 1 - pred[0][0]
    else:  # softmax
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = pred[0][predicted_class]

    return {class_names[predicted_class]: float(confidence)}

# Create Gradio Interface
# -----------------------

import gradio as gr

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI vs Real Image Detector",
    description="Upload an image to check whether it is AI-Generated or Real."
)

interface.launch(share=True)
