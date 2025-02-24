import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2  # Import OpenCV
from PIL import Image  # Import Pillow for image processing

# Load your trained FER-2013 model (replace with your model path)
model = tf.keras.models.load_model('/content/best_modelV4.h5')

# Emotion labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(img_array, target_size=(48, 48)):
    """Predicts emotion from a preprocessed image array."""
    try:
        img = cv2.resize(img_array, target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

    try:
        predictions = model.predict(img)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    predicted_class = np.argmax(predictions)
    score = predictions[0][predicted_class]

    return predicted_class, score

def main():
    st.title("Facial Emotion Recognition")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file)
            img_array = np.array(image_pil)

            if len(img_array.shape) == 3 and img_array.shape[2] == 3: #check if color image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) #convert to grayscale
            elif len(img_array.shape) != 2: # handle images with alpha channel, or other problems.
                st.error("Image has an unsupported number of channels.")
                return;

        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        result = predict_emotion(img_array)

        if result:
            predicted_class, score = result
            st.image(img_array, caption="Uploaded Image", use_column_width=True)
            st.write(f"Predicted Emotion: {class_names[predicted_class]}")
            st.write(f"Confidence: {100 * score:.2f}%")
        else:
            st.write("Prediction failed.")

if __name__ == "__main__":
    main()