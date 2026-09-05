import os
import warnings
import json
import pickle
from collections import OrderedDict

# --- FLASK IMPORTS ---
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup

# ==========================================
# ⚠️ CRITICAL IMPORT ORDER FIX ⚠️
# PyTorch MUST be imported and initialized BEFORE TensorFlow to avoid cuDNN version conflicts.
# ==========================================

# --- VIDEO/IMAGE MODEL IMPORTS (PyTorch) ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models # Added for Image Model architecture
from PIL import Image
import cv2

# Check/Init PyTorch CUDA first
if torch.cuda.is_available():
    video_device = torch.device("cuda")
    # Force CUDA initialization
    torch.ones(1).to(video_device)
    print(f"✅ PyTorch GPU Initialized: {torch.cuda.get_device_name(0)}")
else:
    video_device = torch.device("cpu")
    print("❌ PyTorch running on CPU")

# Dependencies
# pip install facenet-pytorch timm
try:
    from facenet_pytorch import MTCNN 
    import timm 
except ImportError:
    print("⚠️ Warning: 'facenet-pytorch' or 'timm' not found. Video features may fail.")

# --- TEXT/AUDIO MODEL IMPORTS (TensorFlow) ---
# Now we import TensorFlow (it will use the CUDA context PyTorch established, or handle itself)
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore 
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore 
import numpy as np
import librosa

# ==========================================
# CONFIGURATION
# ==========================================

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'flv'}
ALLOWED_EXTENSIONS_AUDIO = {'mp3', 'wav', 'ogg', 'm4a', 'opus'}
ALLOWED_EXTENSIONS_HTML = {'html', 'htm'}
ALLOWED_EXTENSIONS_IMAGE = {'jpg', 'jpeg', 'png', 'webp'} 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

# --- Model File Paths ---
TEXT_WEIGHTS_FILE = "models/best_ai_vs_human_detector.h5"
TEXT_ENCODER_FILE = "label_encoder.pkl"
TEXT_TOKENIZER_FILE = "tokenizer.json"
AUDIO_WEIGHTS_FILE = "models/audio_detector.h5" 
VIDEO_MODEL_FILE = "models/best_deepfake_model.pth" 
IMAGE_MODEL_FILE = "models/full_ai_detector_model.pt" # New Image Model File

# --- Constants ---
MAX_LEN_TEXT = 150

# Audio Constants
AUDIO_IMG_HEIGHT = 224 
AUDIO_IMG_WIDTH = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 4 

# Video Constants
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 224
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2

# ==========================================
# 1. SETUP & GLOBAL LOADERS
# ==========================================

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# --- GPU Check (TensorFlow) ---
print("--- GPU Setup (TF) ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ TensorFlow GPU: {gpus[0].name}")
    except RuntimeError:
        pass
else:
    print("❌ TensorFlow running on CPU")


# --- LOAD TEXT MODEL ---
print("\n--- Loading Text Model ---")
try:
    text_model = tf.keras.models.load_model(TEXT_WEIGHTS_FILE)
    with open(TEXT_ENCODER_FILE, 'rb') as f:
        text_encoder = pickle.load(f)
    with open(TEXT_TOKENIZER_FILE, 'r', encoding='utf-8') as f:
        text_tokenizer = tokenizer_from_json(json.load(f))
    
    text_prob_indices = {
        'Human': text_encoder.transform(['Human-written'])[0],
        'ChatGPT': text_encoder.transform(['AI-generated-chatGPT'])[0],
        'Gemini': text_encoder.transform(['AI-generated-Gemini'])[0]
    }
    print("✅ Text Model Loaded.")
except Exception as e:
    print(f"❌ Error loading Text Model: {e}")


# --- LOAD AUDIO MODEL ---
print("\n--- Loading Audio Model ---")
def build_audio_architecture():
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(AUDIO_IMG_HEIGHT, AUDIO_IMG_WIDTH, 3),
            include_top=False,
            weights=None 
        )
        base_model.trainable = False 

        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=(AUDIO_IMG_HEIGHT, AUDIO_IMG_WIDTH, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3), 
            tf.keras.layers.Dense(1, activation='sigmoid') 
        ])
        return model
    except Exception as e:
        print(f"Error building audio architecture: {e}")
        return None

try:
    audio_model = build_audio_architecture()
    if audio_model:
        audio_model.load_weights(AUDIO_WEIGHTS_FILE)
        print(f"✅ Audio Model Loaded (Weights) from '{AUDIO_WEIGHTS_FILE}'.")
    else:
        print("❌ Failed to build Audio Model architecture.")
except Exception as e:
    print(f"❌ Error loading Audio Model: {e}")


# --- LOAD VIDEO MODEL (CNN + LSTM) ---
print("\n--- Loading Video Model (CNN + LSTM) ---")

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity() 
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=0.4
        )
        
        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        features = self.backbone(x) 
        features = features.view(b, seq, -1)
        lstm_out, _ = self.lstm(features)
        last_frame_out = lstm_out[:, -1, :]
        return self.fc(last_frame_out)

mtcnn = None
try:
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE, margin=20, 
        keep_all=False, select_largest=True, 
        post_process=False, device=video_device
    )
    print("✅ MTCNN Face Detector Loaded.")
except Exception as e:
    print(f"❌ Error loading MTCNN: {e}")

video_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_model = None
try:
    model_instance = DeepfakeDetector()
    if os.path.exists(VIDEO_MODEL_FILE):
        state_dict = torch.load(VIDEO_MODEL_FILE, map_location=video_device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v     
        model_instance.load_state_dict(new_state_dict)
        model_instance.to(video_device)
        model_instance.eval()
        video_model = model_instance
        print(f"✅ Video Model Loaded from '{VIDEO_MODEL_FILE}'.")
    else:
        print(f"❌ Video Model file '{VIDEO_MODEL_FILE}' not found.")
except Exception as e:
    print(f"❌ Error loading Video Model: {e}")


# --- LOAD IMAGE MODEL (ConvNeXt-Tiny) ---
print("\n--- Loading Image Model ---")

# 1. Image Transforms (Same as training code)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 2. Load Model
image_model = None
try:
    if os.path.exists(IMAGE_MODEL_FILE):
        # Loading the full model object as saved in the training script
        # Using map_location to ensure it loads to the correct device (CPU/GPU)
        try:
             # Try with weights_only=False first (as per your script)
             image_model = torch.load(IMAGE_MODEL_FILE, map_location=video_device, weights_only=False)
        except TypeError:
             # Fallback for older torch versions that don't support weights_only
             image_model = torch.load(IMAGE_MODEL_FILE, map_location=video_device)
             
        image_model.eval()
        print(f"✅ Image Model Loaded from '{IMAGE_MODEL_FILE}'.")
    else:
        print(f"❌ Image Model file '{IMAGE_MODEL_FILE}' not found.")
except Exception as e:
    print(f"❌ Error loading Image Model: {e}")


# ==========================================
# 2. PREDICTION FUNCTIONS
# ==========================================

# --- TEXT CHECK ---
def check_ai_text(text_content):
    print(f"Analyzing Text: {text_content[:30]}...")
    input_texts = [text_content]
    sequences = text_tokenizer.texts_to_sequences(input_texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN_TEXT, padding='post', truncating='post')

    try:
        probabilities = text_model.predict(padded_sequences, verbose=0)[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_label = text_encoder.inverse_transform([predicted_class_index])[0]
        
        prob_str = (f"Human: {probabilities[text_prob_indices['Human']]:.4f} | "
                    f"ChatGPT: {probabilities[text_prob_indices['ChatGPT']]:.4f} | "
                    f"Gemini: {probabilities[text_prob_indices['Gemini']]:.4f}")
        
        return f"Result: {predicted_label} (Confidence: {probabilities[predicted_class_index]:.2%})<br>Details: {prob_str}"
    except Exception as e:
        return f"Error: {e}"


# --- AUDIO CHECK ---
def preprocess_audio(audio_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION)
            if y is None or len(y) == 0: return None
            
            target_length = AUDIO_SAMPLE_RATE * AUDIO_DURATION
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            
            min_val = np.min(log_S)
            max_val = np.max(log_S)
            if max_val != min_val:
                norm_S = (log_S - min_val) / (max_val - min_val)
            else:
                norm_S = np.zeros(log_S.shape)
            
            img = (norm_S * 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img_resized = cv2.resize(img, (AUDIO_IMG_WIDTH, AUDIO_IMG_HEIGHT))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            return img_rgb
            
        except Exception as e:
            print(f"Audio Error: {e}")
            return None

def check_ai_audio(filepath):
    print(f"Analyzing Audio: {filepath}")
    img_array = preprocess_audio(filepath)
    if img_array is None: return "Error processing audio."
    
    img_batch = np.expand_dims(img_array, axis=0)
    
    try:
        pred = audio_model.predict(img_batch, verbose=0)[0][0]
        if pred > 0.5:
            label = "Real (Human)"
            confidence = pred
        else:
            label = "Fake (AI)"
            confidence = 1.0 - pred 
        return f"Result: {label} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"Error: {e}"


# --- VIDEO CHECK ---
def check_deepfake_video(filepath):
    print(f"Analyzing Video: {filepath}")
    if video_model is None: return "Error: Video model not loaded."
    if mtcnn is None: return "Error: MTCNN face detector not loaded."
    
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): return "Error: Cannot open video."
        
        frames = []
        frame_count = 0
        stride = 2 
        
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % stride == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    face = mtcnn(frame_rgb)
                    if face is not None:
                        face_img = face.permute(1, 2, 0).cpu().numpy().astype('uint8')
                        pil_img = Image.fromarray(face_img)
                        tensor_img = video_transform(pil_img)
                        frames.append(tensor_img)
                except:
                    pass
            frame_count += 1
            if frame_count > 1500 and len(frames) == 0: break
                
        cap.release()

        if len(frames) == 0:
            return "Error: No faces detected in the first segment."

        while len(frames) < SEQUENCE_LENGTH:
            frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))

        input_batch = torch.stack(frames[:SEQUENCE_LENGTH]).unsqueeze(0).to(video_device)
        
        print(f"   -> Running inference on sequence of {SEQUENCE_LENGTH} frames...")
        
        with torch.no_grad():
            logit = video_model(input_batch)
            prob = torch.sigmoid(logit).item()
            
        if prob > 0.5:
            label = "Real Video"
            confidence = prob
        else:
            label = "Deepfake Detected"
            confidence = 1.0 - prob
            
        return f"Result: {label} (Confidence: {confidence:.2%})"
        
    except Exception as e:
        print(f"Video Error: {e}")
        return f"Error: {e}"


# --- IMAGE CHECK (NEW) ---
def check_ai_image(filepath):
    """
    Checks if an image is Real or AI-generated using ConvNeXt-Tiny.
    Based on the provided training code logic.
    """
    print(f"Analyzing Image: {filepath}")
    if image_model is None: return "Error: Image model not loaded."
    
    try:
        # 1. Load and Convert Image
        img = Image.open(filepath).convert('RGB')
        
        # 2. Transform (Resize, ToTensor, Normalize) and add Batch Dimension
        # The .to(video_device) ensures it goes to GPU if available
        img_t = image_transforms(img).unsqueeze(0).to(video_device)
        
        # 3. Predict
        with torch.no_grad():
            output = image_model(img_t)
            prob = output.item()
            
        # 4. Interpret Result (Threshold 0.5)
        if prob > 0.5:
            label = "Real"
            confidence = prob
        else:
            label = "Fake (AI)"
            confidence = 1.0 - prob
            
        return f"Result: {label} (Confidence: {confidence:.2%})"
        
    except Exception as e:
        print(f"Image Error: {e}")
        return f"Error analyzing image: {e}"


# --- HTML CHECK ---
def analyze_html_content(filepath):
    print(f"Analyzing HTML: {filepath}")
    results = []
    base_dir = os.path.dirname(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        feed_div = soup.find('div', class_='feed')
        if not feed_div: return [{"id": 0, "error": "No <div class='feed'> found."}]
        
        posts = feed_div.find_all('div', class_='post', recursive=False)
        
        for i, post in enumerate(posts):
            post_data = {'id': i + 1, 'content_type': [], 'analysis': [], 'preview': []}

            text_p = post.find('p')
            if text_p:
                text_str = text_p.get_text(strip=True)
                if text_str:
                    post_data['content_type'].append('Text')
                    post_data['preview'].append(f"Text: {text_str[:50]}...")
                    post_data['analysis'].append(f"Text: {check_ai_text(text_str)}")

            media = post.find(['video', 'audio', 'img'])
            if media and media.get('src'):
                src = media.get('src')
                clean_src = src.lstrip('./').lstrip('/')
                abs_path = os.path.normpath(os.path.join(base_dir, clean_src))
                
                post_data['preview'].append(f"Media: {src}")
                
                if not os.path.exists(abs_path):
                    post_data['analysis'].append(f"Media: File not found on server ({src})")
                elif media.name == 'video':
                    post_data['content_type'].append('Video')
                    post_data['analysis'].append(f"Video: {check_deepfake_video(abs_path)}")
                elif media.name == 'audio':
                    post_data['content_type'].append('Audio')
                    post_data['analysis'].append(f"Audio: {check_ai_audio(abs_path)}")
                elif media.name == 'img':
                    post_data['content_type'].append('Image')
                    post_data['analysis'].append(f"Image: {check_ai_image(abs_path)}") # UPDATED

            post_data['content_type'] = ' + '.join(post_data['content_type']) or "Unknown"
            post_data['preview'] = ' | '.join(post_data['preview'])
            post_data['analysis'] = '<br>'.join(post_data['analysis']) or "No analyzable content."
            results.append(post_data)
    except Exception as e:
        return [{"id": 0, "error": str(e)}]
    return results


# ==========================================
# 3. FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/video_check', methods=['GET', 'POST'])
def video_check():
    result = None
    if request.method == 'POST':
        if 'video_file' not in request.files: return redirect(request.url)
        file = request.files['video_file']
        if file.filename == '': return redirect(request.url)
        
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VIDEO):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = check_deepfake_video(filepath)
    return render_template('video_check.html', result=result)

@app.route('/audio_check', methods=['GET', 'POST'])
def audio_check():
    result = None
    if request.method == 'POST':
        if 'audio_file' not in request.files: return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '': return redirect(request.url)
        
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_AUDIO):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = check_ai_audio(filepath)
    return render_template('audio_check.html', result=result)

@app.route('/text_check', methods=['GET', 'POST'])
def text_check():
    result = None
    text_content = ""
    if request.method == 'POST':
        text_content = request.form.get('text_content', '')
        if text_content:
            result = check_ai_text(text_content)
    return render_template('text_check.html', result=result, submitted_text=text_content)

@app.route('/page_check', methods=['GET', 'POST'])
def page_check():
    results = None
    html_preview_url = None
    
    if request.method == 'POST':
        if 'html_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['html_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_HTML):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            html_preview_url = url_for('uploaded_file', filename=filename)
            results = analyze_html_content(filepath)
        else:
            flash('Invalid file type. Please upload an HTML file.')
            return redirect(request.url)

    return render_template('page_check.html', results=results, html_preview_url=html_preview_url)

@app.route('/image_check', methods=['GET', 'POST'])
def image_check():
    result = None
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('No image file part')
            return redirect(request.url)
        file = request.files['image_file']
        if file.filename == '':
            flash('No selected image file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                result = check_ai_image(filepath) 
            except Exception as e:
                 flash(f'An error occurred processing the file: {e}')
                 return redirect(request.url)
        else:
             flash('Invalid file type for image.')
             return redirect(request.url)
    return render_template('image_check.html', result=result)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='127.0.0.1')