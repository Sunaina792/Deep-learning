import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
import cv2

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("✍️ Draw a Digit (0–9) and Let the Model Predict")

# Load the trained model
model = load_model("mnist_model.keras")

st.markdown("Draw a digit in the box below. The model will predict it in real time.")

# Canvas settings
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=12,
    stroke_color="#FFFFFF",  # White brush (digit)
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict if something was drawn
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Convert RGBA to grayscale
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Resize to 28x28 (as required by MNIST model)
    resized = cv2.resize(gray, (28, 28))

    # Normalize and reshape for model
    processed = resized / 255.0
    processed = processed.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(processed)
    pred_class = np.argmax(prediction)

    st.write("### Prediction:")
    st.success(f"🧠 The model predicts: **{pred_class}**")

    st.write("### Model Confidence:")
    st.bar_chart(prediction[0])
