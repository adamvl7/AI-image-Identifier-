"""How the MobileNetV2 model works is that we pass it an image and then it breaks
the image down in arrays of numbers and then each number is a percentage of how close
it is to a certain object. Then after it gets decoded by the decoder so that its not 
a bunch of numbers but actual words that we can understand. And then it grabs the 
first response with the top 3."""

import cv2 
import numpy as np
import streamlit as st

"""tensor flow and keras are pre trained libraries for image classification. 
thats why we import them in. And for model we use mobilenetv2 which is a 
pre trained model. Its lightweight and efficient for mobile and web applications."""
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


def load_model():# Load the pre-trained MobileNetV2 model
    model = MobileNetV2(weights="imagenet") # MobileNetV2 is a convolutional neural network. The weights are a learned value to make the model work how it works. There is different weights.
    return model

def preprocess_image(image): 
    img = np.array(image) # Convert the image to a numpy array
    img = cv2.resize(img, (224, 224)) # Resize the image to 224x224 pixels
    img = preprocess_input(img) # Preprocess the image for MobileNetV2
    img = np.expand_dims(img, axis=0) # Add a batch dimension so it looks like its multiple images.
    return img

def classify_image(model, image):
    try: 
        preprocessed_image = preprocess_image(image) # Takes the preprocessed image from the function above
        predictions = model.predict(preprocessed_image) # Predict the image
        decoded_predictions = decode_predictions(predictions, top=3)[0] # Decode the predictions to get the top 3 predictions, the 0 index is for the first image in the batch.
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout ="centered")

    st.title("AI Image Classifier 🤖")
    st.write("Upload an image and let the AI classify it for you!")

    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption= "Uploaded Image.", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Classifying..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")
                         

if __name__ == "__main__":
    main()
                    
