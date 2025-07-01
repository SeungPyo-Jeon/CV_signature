import streamlit as st
from main import load_img, create_index, search_index
import torch
import os
from PIL import Image
from io import StringIO
import numpy as np
import faiss
import json

def main():
    st.title("Image Retrieval with DINOv2")
    
    cwd = os.getcwd()
    root_dir = os.path.join(cwd, "COCO-128-2/train")
    files = os.listdir(root_dir)
    files = [ os.path.join(root_dir, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')) ]
    
    # Load the model
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    
    data_index = faiss.read_index("data.bin")
    input_image = st.file_uploader("Upload an Image to Search", type=["jpg", "jpeg", "png"])

    print( data_index.ntotal, data_index.d )
    if input_image:
        st.write("Input image:")
        col = st.columns(3)
        with col[0]:
            st.image(input_image)

        with torch.no_grad():
            input_embedding = dinov2_vits14(load_img(input_image).to(device))
            results = search_index(data_index, input_embedding[0].cpu().numpy().reshape(1, -1), k=3)
        
        st.write("Search Results:")
        col = st.columns(3)
        for i, index in enumerate(results):
            with col[i]:
                st.image(files[index], caption=f"Result {i+1}", use_container_width=True)

if __name__ == "__main__":
    main()