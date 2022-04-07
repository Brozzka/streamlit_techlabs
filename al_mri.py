import streamlit as st
import numpy as np # linear algebra
from keras.models import load_model

import os
from PIL import Image, ImageOps
import cv2

W = 224
H = 224
#168

st.header("Alzheimer's Disease Prediction")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the patient's MRI image.")
st.write("This application uses VGG16")

root_dir = os.path.dirname(__file__)
model_path = os.path.join(root_dir,'vgg')
vgg16 = load_model(model_path)


file = st.file_uploader("Please upload an mri image.", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (W, H)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("No image file has been uploaded.")
else:
    image = Image.open(file)
    predictions = import_and_predict(image, vgg16)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)