import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="fine_tuned_xception.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
CLASS_NAMES = ["Bike", "Car"]

# Streamlit UI
st.title("Car vs Bike Image Classifier")

# Choose Upload Method
genre = st.radio("Choose Image Source", ("Upload", "Camera"))

# Get Image
image_file = None
if genre == "Camera":
    image_file = st.camera_input("Capture an Image")
else:
    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    try:
        # Open and display image
        image = Image.open(image_file)
        st.image(image, caption="Selected Image", use_container_width=True)

        # "Predict" button to trigger inference
        if st.button("Predict"):
            # Preprocess image
            image = image.resize((224, 224))  # Resize to match model input size
            image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]

            # Ensure image has 3 channels (Convert grayscale to RGB)
            if image.shape[-1] == 1:  
                image = np.repeat(image, 3, axis=-1)  # Convert grayscale to RGB

            # Expand dimensions to match model input shape (1, 224, 224, 3)
            image = np.expand_dims(image, axis=0)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], image)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Process predictions
            predicted_index = np.argmax(output_data)  # Get highest probability index
            predicted_class = CLASS_NAMES[predicted_index]  # Get class label
            confidence = np.max(output_data) * 100  # Convert to percentage

            # Display Prediction
            st.header(f"Prediction: **{predicted_class}**")
            st.subheader(f"Confidence: **{confidence:.2f}%**")

    except UnidentifiedImageError:
        st.error("Invalid Image Format! Please upload a valid JPG, JPEG, or PNG file.")
