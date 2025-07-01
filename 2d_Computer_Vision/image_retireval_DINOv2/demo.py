import streamlit as st
from main import load_img, create_index, search_index
import torch
import os
from PIL import Image
from io import StringIO
import numpy as np
import faiss
import json
import pandas as pd
from streamlit_image_select import image_select


def display_layout( train_files, test_files ):
    
    col = st.columns([10.9, 0.2, 10.9])
    selected_img = None

    with col[0]:
        st.write("## Test images  *(select an image to search)*")
        with st.spinner("Loading images..."):
            result_container = st.container( height=600, border=True)
            if 'selected_image' not in st.session_state:
                st.session_state.selected_image = None
            with result_container:
                selected_img = image_select(
                    label="Select an image",
                    images=test_files,
                    use_container_width=True,
                    index=0, return_value='index'
                )
            input_image = st.file_uploader("Upload an Image to Search", type=["jpg", "jpeg", "png"])

    with col[1]:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.9);
                        height: 600px;
                        margin: auto;
                    }
                </style>
            '''
        )
    with col[2]:
        st.session_state.selected_container = st.container( height=0, border=True)
        with st.spinner("Loading images..."):
            st.write("## Images in DB")
            result_container = st.container( height=300, border=True)
            recognition_result_container = result_container.columns(10)

            for i, f in enumerate(train_files):
                recognition_result_container[i % 10].image(f, use_container_width=True)
    return selected_img, input_image

def display_smiliar_images( input_image, train_files, data_index, dinov2_vits14 , device="cpu" ):
    
    with torch.no_grad():
        input_embedding = dinov2_vits14(load_img(input_image).to(device))
        results = search_index(data_index, input_embedding[0].cpu().numpy().reshape(1, -1), k=3)
    
    with st.session_state.selected_container:
        st.write("## Search Results in DB")
        #st.height(300)
        cols = st.columns(3)
        for i, index in enumerate(results):
            cols[i % 3].image(train_files[index], caption=f"Result {i+1}", use_container_width=True)

def main():
    st.title("Image Retrieval with DINOv2")
    st.set_page_config(layout="wide")
    cwd = os.getcwd()
    root_dir = os.path.join(cwd, "2d_Computer_Vision/image_retireval_DINOv2/COCO-128-2/train")
    train_files = os.listdir(root_dir)
    train_files = [ os.path.join(root_dir, f) for f in train_files if f.lower().endswith(('.jpg', '.jpeg', '.png')) ]
    
    root_dir = os.path.join(cwd, "2d_Computer_Vision/image_retireval_DINOv2/COCO-128-2/test")
    test_files = os.listdir(root_dir)
    test_files = [ os.path.join(root_dir, f) for f in test_files if f.lower().endswith(('.jpg', '.jpeg', '.png')) ]
    
    selected_img, input_image = display_layout(train_files, test_files)

    # Load the model
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)
    
    data_index = faiss.read_index(os.path.join(cwd,"2d_Computer_Vision/image_retireval_DINOv2/data.bin"))
    
    if selected_img:
        print(f"Selected image index: {selected_img}")

        print(f"state image index: {st.session_state.selected_image}")
        new_index = selected_img
        if new_index != st.session_state.selected_image:
            st.session_state.selected_image = new_index
            display_smiliar_images(test_files[new_index], train_files, data_index, dinov2_vits14, device)

    if input_image:
        st.write("Input image:")
        col = st.columns(3)
        with col[0]:
            st.image(input_image)

        with torch.no_grad():
            input_embedding = dinov2_vits14(load_img(input_image).to(device))
            results = search_index(data_index, input_embedding[0].cpu().numpy().reshape(1, -1), k=3)
        
        with st.session_state.selected_container:
            st.write("## Search Results in DB")
            #st.height(300)
            cols = st.columns(3)
            for i, index in enumerate(results):
                cols[i % 3].image(train_files[index], caption=f"Result {i+1}", use_container_width=True)

if __name__ == "__main__":
    main()
