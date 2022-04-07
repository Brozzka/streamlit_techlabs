from keras.models import load_model
import streamlit as st
import numpy as np # linear algebra
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
import keras
# load model

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

def load_model():
    model = keras.models.Sequential()
    # load model
    VGG = VGG16(include_top=False, input_shape=(W,H,3),pooling='avg')    
    print(VGG.summary())
    # Freezing Layers

    for layer in VGG.layers:
        layer.trainable=False
        
    model.add(VGG)
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(4,activation='softmax'))
    model.load_weights(os.path.join(os.path.dirname(__file__),'vgg.h5'))
    return model


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