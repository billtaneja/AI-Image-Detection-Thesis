from tensorflow import keras

# Loading trained model
model = keras.models.load_model("/Users/yashtaneja/Desktop/Yash_Thesis/coding and model/AI_vs_Real_Model.h5")

class_names = ["Real", "AI-Generated"]

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

def predict_image(img):
   
    img = img.convert("RGB")

    img = img.resize((224, 224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    # Predictiing
    pred = model.predict(img_array)

    if pred.shape[1] == 1:  
        predicted_class = int(pred[0][0] > 0.5)
        confidence = pred[0][0] if predicted_class == 1 else 1 - pred[0][0]
    else: 
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = pred[0][predicted_class]

    return {class_names[predicted_class]: float(confidence)}

# Gradio!!!
import gradio as gr
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI vs Real Image Detector",
    description="Upload an image to check whether it is AI-Generated or Real.",
    allow_flagging="never"  
)

interface.launch(share=True)
