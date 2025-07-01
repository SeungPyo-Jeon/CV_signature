import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import faiss
from tqdm import tqdm
import numpy as np
import json

transform_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

def load_img( img_path ):
    print(f"Loading image: {img_path}")

    input_img = Image.open(img_path)#.convert("RGB")
    transformed_img = transform_img(input_img)[:3].unsqueeze(0)  # Add batch dimension
    return transformed_img

def create_index(files, model):
    index = faiss.IndexFlatL2(384)  # DINOv2 feature dimension is 384
    all_embeddings = {}
    with torch.no_grad():
        for _, file in enumerate(tqdm(files)):
            img = load_img(file).to(device)
            embedding = model(img)[0].cpu().numpy()
            np_embedding = np.array(embedding).reshape(1, -1)
            all_embeddings[file] = np_embedding.reshape(1,-1).tolist()
            index.add(np_embedding)
    
    with open("all_embeddings.json", "w") as f:
        f.write( json.dumps(all_embeddings) )
    
    faiss.write_index(index, "data.bin")

    return index, all_embeddings

def search_index( input_index, input_embedding, k=3):
    distances, indices = input_index.search( np.array( input_embedding[0].reshape( 1, -1)), k)
    return indices[0]

if __name__ == "__main__":
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    root_dir = os.path.join(cwd, "COCO-128-2/train")

    files = os.listdir(root_dir)
    files = [ os.path.join(root_dir, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')) ]

    dinov2_vits14 = torch.hub.load( "facebookresearch/dinov2", "dinov2_vits14")#, pretrained=True, force_reload=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vits14.to(device)

    data_index, all_embeddings = create_index(files, dinov2_vits14)

    input_file = r"COCO-128-2\valid\000000000081_jpg.rf.5262c2db56ea4568d7d32def1bde3d06.jpg"
    input_img = cv2.resize(cv2.imread(input_file), (416, 416))
    
    with torch.no_grad():
        embedding = dinov2_vits14( load_img(input_file).to(device) )
        results = search_index(data_index, embedding[0].cpu().numpy().reshape(1,-1), k=3)

        for i, index in enumerate(results):
            print(f"Image {i}: {files[index]}")