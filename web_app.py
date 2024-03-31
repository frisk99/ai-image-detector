from io import StringIO
from pathlib import Path
import streamlit as st
import time
import os
import sys
import argparse
from PIL import Image
import argparse
import imageio.v3 as iio
import torch
from PIL import Image
from model import get_model
from custom_dataset import get_transform
import glob
import os
import cv2


def predict_single_image(image, model, device, transform):
    # Load and transform the image
    image = Image.fromarray(image).convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Make a prediction
    model.eval()
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = outputs.logits.max(1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Map numeric labels to string labels
    label_map = {0: "人工", 1: "ai合成"}
    predicted_label = label_map[predicted.item()]

    return predicted_label, probabilities


def load_latest_model(model, device, weights_folder):
    # Use the specified folder or the default path to find the latest model
    list_of_files = glob.glob(os.path.join(weights_folder, 'model_epoch_*.pth'))
    if not list_of_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")
    list_of_bests = glob.glob(os.path.join(weights_folder, '*best.pth'))
    latest_file = max(list_of_files, key=os.path.getctime)
    best_file = max(list_of_bests, key=os.path.getctime)
    if best_file is not None:
        checkpoint = torch.load(best_file, map_location=device)
        print("aaa")
        print(best_file)
    else:
        checkpoint = torch.load(latest_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


@st.cache_resource
def init():
    weights_folder = './models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model = load_latest_model(model, device, weights_folder)
    transform = get_transform()
    return model, device, transform
if __name__ == "__main__":
    model, device, transform = init()
    st.title("AI图片检测")
    image_files = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'],accept_multiple_files=True)
    if image_files is not None:
        for image_file in image_files:
            bytes_data = image_file.read()
            img = iio.imread(image_file)
            # consume img
            st.image(img, channels="RGB")
            predicted_label, probabilities =predict_single_image(img, model, device, transform)
            print(f'Predicted label: {predicted_label}')
            tmp =  1 if predicted_label== "ai合成" else 0
            print(f'Class probabilities: {probabilities.cpu()[0][tmp]}')
            st.write('%f' % float(probabilities.cpu().numpy()[0][tmp]*100)+f'%的概率为{predicted_label}'  )
        st.write('检测结果未必准确 仅供参考')
        st.write('该检测器仅使用artifact训练集训练，可能会有一定误差')
