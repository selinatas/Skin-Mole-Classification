import streamlit as st
from PIL import Image
import torch
import main

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit app
st.title("Skin Mole Classification - Selin Ataş")
st.write("Please upload an image and select a model to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
model_option = st.selectbox("Select model", ["ResNet-50", "DenseNet-121", "Custom CNN"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    if model_option == "ResNet-50":
        class_name = main.predict_image(main.resnet, image, device)
    elif model_option == "DenseNet-121":
        class_name = main.predict_image(main.densenet, image, device)
    else:
        class_name = main.predict_image(main.model, image, device)

    st.write(f"Predicted Class: {class_name}")
