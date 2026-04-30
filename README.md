# Fake Multimedia Content Detection System

![System Overview](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg)

## Overview
As artificial intelligence advances, the generation of hyper-realistic forged content—"Deepfakes"—poses a significant threat to digital integrity. This project presents a comprehensive, four-modality detection system designed to identify AI-generated manipulations across video, text, audio, and images. By leveraging a "fighting AI with AI" approach, this system provides a robust defensive shield against digital misinformation.

This project was developed as a graduation requirement at **Tafila Technical University** (Fall 2025-2026).

**Development Team:** Abdullah Mohammad Hashem Tayeh, Noor Zeyad Saleem Barahmeh, Dana Omar Saleh Abdulqader, Saleh Abdeljabbar Subhi Mohammad  
**Supervisor:** Dr. Rasha Al-Bashaireh  

---

## Modalities & Methodologies

Our system analyzes media through four distinct pipelines, applying specialized deep learning architectures to each domain:

### 1. Video Deepfake Detection
* **Architecture:** EfficientNet-B0 + LSTM
* **Methodology:** Treats video analysis as a temporal sequence problem rather than isolated frame classification. Uses MTCNN for offline facial extraction and alignment. EfficientNet-B0 extracts spatial features from a 20-frame sequence, which are then passed to an LSTM layer to detect temporal anomalies (e.g., unnatural blinking or lip sync).
* **Accuracy:** 93.87%

### 2. AI Text Detection (ChatGPT & Gemini vs. Human)
* **Architecture:** Conv1D + Bi-LSTM (Hybrid)
* **Methodology:** Tokenizes and pads text sequences to a fixed length. The 1D Convolutional layer identifies local, repetitive stylistic patterns typical of LLMs, while the Bidirectional LSTM analyzes long-range semantic connections in both directions to capture the full context.
* **Accuracy:** 98.94%

### 3. Synthetic Audio Detection
* **Architecture:** MobileNetV2
* **Methodology:** Transforms the audio signal processing task into a computer vision problem. Raw audio is standardized, resampled, and converted into Mel-Spectrograms (log-scale). These visual representations are fed into a lightweight CNN (MobileNetV2) to detect unnatural frequency transitions characteristic of synthetic voices.
* **Accuracy:** 97.70%

### 4. AI Image Detection
* **Architecture:** ConvNeXt-Tiny
* **Methodology:** Focuses on pixel-level artifacts and high-frequency relics left behind by GANs and diffusion models. Images are standardized to 224x224 and normalized using ImageNet statistics before being classified by the ConvNeXt architecture.
* **Accuracy:** 99.95%

---

## Tech Stack

* **Deep Learning:** PyTorch, TensorFlow, Keras, Timm
* **Computer Vision:** OpenCV, MTCNN, PIL
* **Audio Processing:** Librosa
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Deployment Interface:** Flask, Gradio (for interactive web applications)

---

## Interface Design

The backend models are wrapped in a user-friendly, responsive Web Application built with Flask. The interface follows an MVC architecture and provides distinct, isolated modules for uploading and verifying Video, Audio, Text, Images, and full HTML web pages. The platform processes inputs securely and returns immediate color-coded confidence scores (Green for Real, Red for Fake).

---

## Acknowledgments
We extend our deepest gratitude to Dr. Rasha Al-Bashaireh for her exceptional supervision, guidance, and support throughout this research. We also thank Tafila Technical University for providing the academic environment necessary to complete this work, as well as our families and friends for their continuous encouragement.
