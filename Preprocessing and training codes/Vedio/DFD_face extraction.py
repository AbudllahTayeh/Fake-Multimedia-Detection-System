# 1. Install Requirements
#install facenet-pytorch  and timm

# 2. Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import os
import glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import timm
import shutil

# 3. Configuration
CONFIG = {
    "seq_length": 20,       # Number of frames per video
    "img_size": 224,        # Target size for CNN
    "batch_size": 16,       # Batch size
    "hidden_size": 128,     # LSTM hidden size
    "num_layers": 2,        # LSTM layers
    "base_model": "efficientnet_b0",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- FACE EXTRACTION LOGIC (Integrated) ---
def preprocess_dfd_videos(video_root, output_root, label_name, stride=5):
    """
    Reads DFD videos, detects faces, and saves them as folders of images.
    Matches the FaceForensics++ structure.
    """
    device = CONFIG['device']
    mtcnn = MTCNN(
        image_size=CONFIG['img_size'], margin=20, 
        keep_all=False, select_largest=True, 
        post_process=False, device=device
    )
    
    # Get all video files
    videos = glob.glob(os.path.join(video_root, "**/*.mp4"), recursive=True)
    print(f"Found {len(videos)} videos in {label_name}...")
    
    for video_path in tqdm(videos):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        save_folder = os.path.join(output_root, label_name, filename)
        
        # Skip if already processed
        if os.path.exists(save_folder) and len(os.listdir(save_folder)) > 5:
            continue
            
        os.makedirs(save_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Process every Nth frame (Stride)
            if frame_count % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    # Detect and crop
                    face_tensor = mtcnn(frame_rgb)
                    if face_tensor is not None:
                        save_path = os.path.join(save_folder, f"{frame_count}.jpg")
                        # Convert tensor to image (0-255)
                        face_img = face_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR) # Save as BGR
                        cv2.imwrite(save_path, face_img)
                        saved_count += 1
                except:
                    pass
            frame_count += 1
        cap.release()
        
        # Cleanup: if video had no faces, remove folder to avoid empty data errors
        if saved_count == 0:
            shutil.rmtree(save_folder)

# --- RUN EXTRACTION ---
# Define where DFD is and where to save processed version
DFD_INPUT_ROOT = "/kaggle/input/deep-fake-detection-dfd-entire-original-dataset"
PROCESSED_ROOT = "/kaggle/working/processed_data"

# 1. Real Videos
real_source = glob.glob(os.path.join(DFD_INPUT_ROOT, "*original*"))[0]
preprocess_dfd_videos(real_source, PROCESSED_ROOT, "real", stride=10)

# 2. Fake Videos
fake_source = glob.glob(os.path.join(DFD_INPUT_ROOT, "*manipulated*"))[0]
preprocess_dfd_videos(fake_source, PROCESSED_ROOT, "fake", stride=10)

print("✅ Phase 1: DFD Face Extraction Complete.")