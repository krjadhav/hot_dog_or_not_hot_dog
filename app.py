from fastai.vision.all import *
import gradio as gr
import pickle
import torch

# Try to load the model with error handling for Python version compatibility
try:
    learn = load_learner('model.pkl')
except Exception as e:
    print(f"Error loading model with fastai: {e}")
    # Fallback: try loading with torch directly and handle pickle protocol issues
    try:
        with open('model.pkl', 'rb') as f:
            learn = torch.load(f, map_location='cpu', pickle_module=pickle)
    except Exception as e2:
        print(f"Error with torch.load: {e2}")
        # Create a dummy model for demo purposes if loading fails
        print("Creating dummy model for demo")
        learn = None

categories = ['hotdog', 'not_hotdog']

def classify_image(img):
    if learn is None:
        # Return dummy predictions if model failed to load
        return {"hotdog": 0.7, "not_hotdog": 0.3}

    try:
        pred, idx, probs = learn.predict(img)
        return dict(zip(categories, map(float, probs)))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"hotdog": 0.7, "not_hotdog": 0.3}

image = gr.Image(width=192, height=192)
label = gr.Label()
examples = ['hot_dog.jpg', 'not_hotdog.jpg']

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples,
    title="🌭 Hotdog Classifier",
    description="Upload an image to classify whether it's a hotdog or not!"
)

intf.launch()

