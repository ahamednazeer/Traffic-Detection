

import streamlit as st
import streamlit.components.v1
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import pyttsx3
import threading
import time
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="YOLO v11 Traffic Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #31333F 0%, #31333F 100%);
        font-family: 'Poppins', sans-serif;
    }
    header {
    visibility: hidden;
    }
    .stActionButton {
    visibility: hidden;
    }
    
    /* Main container with glassmorphism */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 1rem;
    }
    
    /* Animated header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease-in-out infinite;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 300;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out 0.5s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    /* Modern section headers */
    .section-header {
        font-size: 1.8rem;
        color: #FFFFFF;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        position: relative;
        padding-left: 20px;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 30px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border-radius: 2px;
    }
    
    /* Enhanced detection box */
    .detection-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 30px;
        border-radius: 25px;
        margin: 30px 0;
        color: white;
        box-shadow: 0 20px 50px rgba(31, 38, 135, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        animation: slideInUp 0.6s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced stats containers */
    .stats-container {
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.9) 0%, rgba(245, 87, 108, 0.9) 100%);
        backdrop-filter: blur(15px);
        padding: 25px 20px;
        border-radius: 20px;
        margin: 15px 5px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stats-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(240, 147, 251, 0.3);
    }
    
    .stats-container h3 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stats-container p {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        border: none;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Color legend improvements */
    .color-legend {
        background: #31333F;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1px;
        margin: 2px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .color-item {
        display: flex;
        align-items: center;
        padding: 8px 0;
        transition: all 0.3s ease;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    .color-item:hover {
        background: rgba(255, 255, 255, 0.1);
        padding-left: 10px;
    }
    
    .color-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        display: inline-block;
    }
    
    /* Detection stats improvements */
    .detection-stats {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .detection-stats:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(5px);
    }
    
    /* Live detection indicators */
    .live-detection {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(139, 195, 74, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 12px 20px;
        border-radius: 25px;
        margin: 8px 0;
        border: 1px solid rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .live-detection::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .live-detection:hover {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.3) 0%, rgba(139, 195, 74, 0.3) 100%);
        transform: scale(1.02);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Class names from your YOLO model
CLASS_NAMES = [
    "Car", "Pedestrian", "Van", "Cyclist", 
    "Truck", "Misc", "Tram", "Person_sitting"
]

# Color mapping for different classes
CLASS_COLORS = {
    "Car": (255, 0, 0),
    "Pedestrian": (0, 255, 0),
    "Van": (0, 0, 255),
    "Cyclist": (255, 255, 0),
    "Truck": (255, 0, 255),
    "Misc": (0, 255, 255),
    "Tram": (128, 0, 128),
    "Person_sitting": (255, 165, 0)
}

def initialize_tts():
    """Initialize text-to-speech engine"""
    if 'audio_method' not in st.session_state:
        st.session_state.audio_method = 'html'  # Default to HTML audio
    
    if st.session_state.audio_method == 'pyttsx3' and st.session_state.tts_engine is None:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            st.session_state.tts_engine = engine
        except Exception as e:
            st.warning(f"⚠️ pyttsx3 TTS engine could not be initialized: {str(e)}. Using HTML audio instead.")
            st.session_state.audio_method = 'html'

def create_audio_html(text):
    """Create HTML audio element using Web Speech API"""
    audio_html = f"""
    <script>
    function playTTS() {{
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance('{text}');
            utterance.rate = 1.2;
            utterance.volume = 0.8;
            utterance.pitch = 1.0;
            speechSynthesis.speak(utterance);
        }} else {{
            console.log('Speech synthesis not supported');
        }}
    }}
    playTTS();
    </script>
    """
    return audio_html

def speak_detection(text):
    """Play audio alert for detections"""
    if not st.session_state.get('enable_audio', True):
        return
        
    try:
        if st.session_state.get('audio_method') == 'html':
            # Use HTML/JavaScript Web Speech API
            audio_html = create_audio_html(text)
            st.components.v1.html(audio_html, height=0)
        else:
            # Fallback to pyttsx3 in a safer way
            def speak_safe():
                try:
                    if st.session_state.tts_engine:
                        st.session_state.tts_engine.say(text)
                        st.session_state.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
            
            # Use a daemon thread to avoid blocking
            thread = threading.Thread(target=speak_safe, daemon=True)
            thread.start()
    except Exception as e:
        print(f"Audio error: {e}")

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def draw_detections(image, results):
    """Draw bounding boxes and labels on image"""
    detections_info = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                if class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                    color = CLASS_COLORS.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background
                    label = f"{class_name}: {confidence:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        image, (x1, y1 - label_height - 10), 
                        (x1 + label_width, y1), color, -1
                    )
                    cv2.putText(
                        image, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    
                    # Store detection info
                    detections_info.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
    
    return image, detections_info

def process_image(image, model, confidence_threshold):
    """Process single image for detection"""
    results = model(image, conf=confidence_threshold)
    annotated_image, detections = draw_detections(image.copy(), results)
    
    # Speak detections
    if detections:
        detected_classes = list(set([d['class'] for d in detections]))
        pedestrian_detected = any(cls in ['Pedestrian', 'Person_sitting', 'Cyclist'] for cls in detected_classes)
        vehicle_detected = any(cls in ['Car', 'Van', 'Truck', 'Tram'] for cls in detected_classes)
        
        if pedestrian_detected and vehicle_detected:
            speak_detection("Pedestrian and vehicle detected")
        elif pedestrian_detected:
            speak_detection("Pedestrian detected")
        elif vehicle_detected:
            speak_detection("Vehicle detected")
    
    return annotated_image, detections

def display_enhanced_detection_stats(detections, processing_time=None):
    """Display enhanced detection statistics with better UI"""
    if not detections:
        return
    
    st.markdown('<div class="detection-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Detection Analytics")
    
    # Create dataframe
    df = pd.DataFrame(detections)
    
    # Top row - Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
            <div class="stats-container">
                <h3>{len(detections)}</h3>
                <p>Total Objects</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        unique_classes = df['class'].nunique()
        st.markdown(f'''
            <div class="stats-container">
                <h3>{unique_classes}</h3>
                <p>Unique Classes</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_confidence = df['confidence'].mean()
        st.markdown(f'''
            <div class="stats-container">
                <h3>{avg_confidence:.1%}</h3>
                <p>Avg Confidence</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        if processing_time:
            st.markdown(f'''
                <div class="stats-container">
                    <h3>{processing_time:.2f}s</h3>
                    <p>Process Time</p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            highest_conf = df['confidence'].max()
            st.markdown(f'''
                <div class="stats-container">
                    <h3>{highest_conf:.1%}</h3>
                    <p>Highest Conf</p>
                </div>
            ''', unsafe_allow_html=True)
    
    # Class distribution
    st.markdown("#### 🎯 Detection Breakdown")
    class_counts = df['class'].value_counts()
    
    for class_name, count in class_counts.items():
        confidence_for_class = df[df['class'] == class_name]['confidence'].mean()
        color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS.get(class_name, (128, 128, 128))[::-1])
        
        # Progress bar representation
        percentage = (count / len(detections)) * 100
        st.markdown(f'''
            <div class="detection-stats">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center;">
                        <span class="color-dot" style="background-color: {color_hex}; margin-right: 10px;"></span>
                        <strong>{class_name}</strong>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.1em; font-weight: bold;">{count} objects</div>
                        <div style="font-size: 0.9em; opacity: 0.8;">Conf: {confidence_for_class:.1%}</div>
                    </div>
                </div>
                <div style="width: 100%; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 8px; height: 6px;">
                    <div style="width: {percentage}%; background: {color_hex}; height: 6px; border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    # Safety alerts section
    pedestrians = df[df['class'].isin(['Pedestrian', 'Person_sitting', 'Cyclist'])].shape[0]
    vehicles = df[df['class'].isin(['Car', 'Van', 'Truck', 'Tram'])].shape[0]
    
    if pedestrians > 0 and vehicles > 0:
        st.markdown(f'''
            <div style="background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 152, 0, 0.2) 100%); 
                 padding: 15px; border-radius: 15px; margin: 15px 0; 
                 border: 1px solid rgba(255, 193, 7, 0.3);">
                <h4 style="color: #FFC107; margin: 0 0 10px 0;">⚠️ Safety Alert</h4>
                <p style="margin: 0; color: white;">
                    Mixed traffic detected: <strong>{pedestrians} pedestrian(s)</strong> and <strong>{vehicles} vehicle(s)</strong> in the same frame. 
                    Exercise caution in this area.
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_realtime_stats(current_detections):
    """Display real-time detection statistics with enhanced UI"""
    if not current_detections:
        return
    
    st.markdown("### 🔴 Live Detection Feed")
    
    # Group detections by class for better display
    detection_counts = {}
    for detection in current_detections:
        class_name = detection['class']
        if class_name not in detection_counts:
            detection_counts[class_name] = {'count': 0, 'max_conf': 0}
        detection_counts[class_name]['count'] += 1
        detection_counts[class_name]['max_conf'] = max(detection_counts[class_name]['max_conf'], detection['confidence'])
    
    # Display with enhanced styling
    for class_name, data in detection_counts.items():
        color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS.get(class_name, (128, 128, 128))[::-1])
        
        st.markdown(f'''
            <div class="live-detection">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center;">
                        <span class="color-dot" style="background-color: {color_hex}; margin-right: 12px;"></span>
                        <span style="font-weight: 600; font-size: 1.1em;">{class_name}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-weight: bold; font-size: 1.1em;">{data['count']} detected</div>
                        <div style="font-size: 0.9em; opacity: 0.8;">Max: {data['max_conf']:.1%}</div>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

# Main app
def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🚗 YOLO v11 Traffic Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🎯 Real-Time Traffic Sign and Pedestrian Detection for Autonomous Driving</div>', 
                unsafe_allow_html=True)
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Model status indicator
        st.markdown('<div class="color-legend">', unsafe_allow_html=True)
        if st.session_state.model_loaded:
            st.markdown('🟢 **Model Status:** Loaded & Ready')
        else:
            st.markdown('🔴 **Model Status:** Not Loaded')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model path (hardcoded as per your preference)
        model_file = "best (8).pt"
        
        # Settings with improved styling
        st.markdown("### 🎛️ Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Higher values = more confident detections only"
        )
        
        st.markdown("### 🔊 Audio Configuration")
        enable_audio = st.checkbox("🔊 Enable Audio Alerts", value=True)
        st.session_state.enable_audio = enable_audio
        
        if enable_audio:
            audio_method = st.selectbox(
                "Audio Method",
                ["html", "pyttsx3"],
                format_func=lambda x: "🌐 Web Speech API (Recommended)" if x == "html" else "🎵 pyttsx3 (Alternative)",
                index=0,
                help="Web Speech API works directly in browser"
            )
            st.session_state.audio_method = audio_method
            initialize_tts()
        
        # Enhanced class colors display
        st.markdown('<div class="color-legend">', unsafe_allow_html=True)
        st.markdown("### 🎨 Detection Classes")
        for class_name, color in CLASS_COLORS.items():
            color_hex = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            st.markdown(f'''
                <div class="color-item">
                    <span class="color-dot" style="background-color: {color_hex};"></span>
                    <span style="color: white; font-weight: 500;">{class_name}</span>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance metrics
        if st.session_state.model_loaded:
            st.markdown('<div class="color-legend">', unsafe_allow_html=True)
            st.markdown("### 📊 Quick Stats")
            st.markdown("🏁 **Classes:** 8 Types")
            st.markdown("⚡ **Framework:** YOLO v11")
            st.markdown("🎯 **Dataset:** KITTI")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        if os.path.exists(model_file):
            with st.spinner("Loading YOLO model..."):
                model = load_model(model_file)
                if model:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
        else:
            st.error(f"❌ Model file '{model_file}' not found. Please ensure the file exists in the same directory.")
            return
    
    if not st.session_state.model_loaded or st.session_state.model is None:
        st.warning("⚠️ Model could not be loaded. Please check the model file.")
        return
    
    # Detection mode selection with enhanced styling
    st.markdown('<h2 class="section-header">‎ 🎯 Detection Mode</h2>', unsafe_allow_html=True)
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📸 Image Detection", key="image_mode", use_container_width=True):
            st.session_state.detection_mode = "image"
    
    with col2:
        if st.button("🎥 Video Detection", key="video_mode", use_container_width=True):
            st.session_state.detection_mode = "video"
    
    with col3:
        if st.button("📹 Live Camera", key="camera_mode", use_container_width=True):
            st.session_state.detection_mode = "camera"
    
    # Initialize detection mode if not set
    if 'detection_mode' not in st.session_state:
        st.session_state.detection_mode = "image"
    
    # Display current mode
    mode_icons = {"image": "📸", "video": "🎥", "camera": "📹"}
    mode_names = {"image": "Image Detection", "video": "Video Detection", "camera": "Real-time Camera"}
    
    st.markdown(f'''
        <div style="text-align: center; margin: 20px 0; padding: 15px; 
             background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); 
             border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
            <h3 style="color: white; margin: 0;">
                {mode_icons[st.session_state.detection_mode]} 
                Current Mode: {mode_names[st.session_state.detection_mode]}
            </h3>
        </div>
    ''', unsafe_allow_html=True)
    
    # Handle different detection modes
    if st.session_state.detection_mode == "image":
        handle_image_detection(st.session_state.model, confidence_threshold)
    elif st.session_state.detection_mode == "video":
        handle_video_detection(st.session_state.model, confidence_threshold)
    else:
        handle_realtime_detection(st.session_state.model, confidence_threshold)

def handle_image_detection(model, confidence_threshold):
    """Handle image detection with enhanced UI"""
    st.markdown('<h3 class="section-header">‎ 📸 Upload Image for Detection</h3>', unsafe_allow_html=True)
    
    # Create upload area with better styling
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="image_upload",
        help="Upload a high-quality image for best detection results"
    )
    
    if uploaded_file is not None:
        # Create two columns with better spacing
        col1, col2 = st.columns(2, gap="large")
        
        # Original image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        with col1:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 15px;">
                    <h4 style="color: white; font-weight: 600;">📷 Original Image</h4>
                </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption=f"Size: {image.size[0]}x{image.size[1]} pixels")
        
        # Detection button with enhanced styling
        detect_button = st.button(
            "🔍 Start Detection", 
            key="detect_image",
            use_container_width=True,
            help="Click to analyze the image and detect objects"
        )
        
        if detect_button:
            with st.spinner("🔄 Analyzing image... Please wait"):
                start_time = time.time()
                annotated_image, detections = process_image(
                    image_np, model, confidence_threshold
                )
                processing_time = time.time() - start_time
                
                with col2:
                    st.markdown("""
                        <div style="text-align: center; margin-bottom: 15px;">
                            <h4 style="color: white; font-weight: 600;">🎯 Detection Results</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(annotated_image, use_container_width=True, 
                            caption=f"Processed in {processing_time:.2f}s")
                
                # Display enhanced detection statistics
                if detections:
                    display_enhanced_detection_stats(detections, processing_time)
                    
                    # Success message with animation
                    st.success(f"✅ Detection completed! Found {len(detections)} objects in {processing_time:.2f} seconds")
                else:
                    st.info("ℹ️ No objects detected with the current confidence threshold. Try lowering the threshold.")
    else:
        # Placeholder when no image is uploaded
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; 
                 background: rgba(255, 255, 255, 0.05); 
                 border-radius: 20px; border: 2px dashed rgba(255, 255, 255, 0.3);
                 margin: 20px 0;">
                <h3 style="color: rgba(255, 255, 255, 0.8); margin-bottom: 20px;">
                    📸 Ready for Image Detection
                </h3>
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 1.1rem;">
                    Upload an image to start detecting vehicles and pedestrians
                </p>
            </div>
        """, unsafe_allow_html=True)

def handle_video_detection(model, confidence_threshold):
    """Handle video detection with enhanced UI"""
    st.markdown('<h3 class="section-header">🎥 Upload Video for Detection</h3>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_upload",
        help="Upload a video file for frame-by-frame object detection"
    )
    
    if uploaded_video is not None:
        # Display video info
        file_size = len(uploaded_video.getvalue()) / (1024 * 1024)  # Convert to MB
        st.markdown(f'''
            <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; 
                 border-radius: 15px; margin: 15px 0; border: 1px solid rgba(255, 255, 255, 0.2);">
                <h4 style="color: white; margin: 0 0 10px 0;">📋 Video Information</h4>
                <p style="margin: 5px 0; color: rgba(255, 255, 255, 0.9);">
                    <strong>File:</strong> {uploaded_video.name}<br>
                    <strong>Size:</strong> {file_size:.1f} MB
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🎬 Start Video Processing", key="process_video", use_container_width=True):
            process_enhanced_video_file(uploaded_video, model, confidence_threshold)
    else:
        # Placeholder for video upload
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; 
                 background: rgba(255, 255, 255, 0.05); 
                 border-radius: 20px; border: 2px dashed rgba(255, 255, 255, 0.3);
                 margin: 20px 0;">
                <h3 style="color: rgba(255, 255, 255, 0.8); margin-bottom: 20px;">
                    🎥 Ready for Video Processing
                </h3>
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 1.1rem;">
                    Upload a video to analyze multiple frames for object detection
                </p>
                <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.9rem;">
                    Supported formats: MP4, AVI, MOV, MKV
                </p>
            </div>
        """, unsafe_allow_html=True)

def process_enhanced_video_file(video_file, model, confidence_threshold):
    """Process uploaded video file with enhanced UI feedback"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getvalue())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Display video info
    st.markdown(f'''
        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; 
             border-radius: 15px; margin: 20px 0; border: 1px solid rgba(255, 255, 255, 0.2);">
            <h4 style="color: white; margin: 0 0 15px 0;">🎬 Processing Video</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                <div><strong>Duration:</strong> {duration:.1f}s</div>
                <div><strong>FPS:</strong> {fps}</div>
                <div><strong>Frames:</strong> {total_frames}</div>
                <div><strong>Status:</strong> <span style="color: #4CAF50;">Processing...</span></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Create output video path
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get frame dimensions
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            col1, col2 = st.columns(2)
            
            with col1:
                frame_placeholder = st.empty()
            with col2:
                stats_placeholder = st.empty()
        
        all_detections = []
        frame_count = 0
        start_time = time.time()
        
        # Process first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = process_image(
                frame, model, confidence_threshold
            )
            all_detections.extend(detections)
            
            # Write frame
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress, text=f"Processing frame {frame_count}/{total_frames}")
            
            # Update status
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            status_text.write(f"⏱️ Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s | Detections: {len(all_detections)}")
            
            # Display current frame every 30 frames
            if frame_count % 30 == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_count}/{total_frames} ({progress:.1%})",
                    use_container_width=True
                )
                
                # Show real-time stats
                if detections:
                    with stats_placeholder:
                        display_realtime_stats(detections)
        
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        
        # Display completion message
        st.success(f"✅ Video processing completed in {total_time:.1f} seconds!")
        
        # Show processed video
        st.markdown("### 🎥 Processed Video")
        st.video(output_path)
        
        # Display comprehensive statistics
        if all_detections:
            display_enhanced_detection_stats(all_detections, total_time)
        
        # Cleanup
        os.unlink(video_path)
        os.unlink(output_path)

def handle_realtime_detection(model, confidence_threshold):
    """Handle real-time camera detection with enhanced UI"""
    st.markdown('<h3 class="section-header">📹 Real-time Camera Detection</h3>', unsafe_allow_html=True)
    
    # Camera controls with enhanced styling
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📹 Start Camera", key="start_camera", use_container_width=True):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("⏸️ Pause", key="pause_camera", use_container_width=True):
            st.session_state.camera_active = False
    
    with col3:
        if st.button("⏹️ Stop Camera", key="stop_camera", use_container_width=True):
            st.session_state.camera_active = False
    
    # Camera status indicator
    if st.session_state.get('camera_active', False):
        st.markdown(f'''
            <div style="text-align: center; padding: 15px; margin: 20px 0;
                 background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(139, 195, 74, 0.2) 100%);
                 border-radius: 15px; border: 1px solid rgba(76, 175, 80, 0.3);">
                <h4 style="color: #4CAF50; margin: 0;">
                    🔴 LIVE - Camera Active
                </h4>
                <p style="margin: 5px 0 0 0; color: rgba(255, 255, 255, 0.8);">
                    Real-time detection in progress...
                </p>
            </div>
        ''', unsafe_allow_html=True)
        run_enhanced_realtime_detection(model, confidence_threshold)
    else:
        st.markdown(f'''
            <div style="text-align: center; padding: 60px 20px; 
                 background: rgba(255, 255, 255, 0.05); 
                 border-radius: 20px; border: 2px dashed rgba(255, 255, 255, 0.3);
                 margin: 20px 0;">
                <h3 style="color: rgba(255, 255, 255, 0.8); margin-bottom: 20px;">
                    📹 Real-time Detection Ready
                </h3>
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 1.1rem; margin-bottom: 20px;">
                    Click "Start Camera" to begin live object detection
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px;">
                        <strong style="color: #4CAF50;">🎯 Live Detection</strong><br>
                        <span style="color: rgba(255, 255, 255, 0.7);">Real-time object recognition</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px;">
                        <strong style="color: #FF9800;">🔊 Audio Alerts</strong><br>
                        <span style="color: rgba(255, 255, 255, 0.7);">Instant voice notifications</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px;">
                        <strong style="color: #2196F3;">📊 Live Stats</strong><br>
                        <span style="color: rgba(255, 255, 255, 0.7);">Real-time analytics</span>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

def run_enhanced_realtime_detection(model, confidence_threshold):
    """Run enhanced real-time detection from camera"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access camera. Please check your camera permissions.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create layout for real-time display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        stats_container = st.container()
        with stats_container:
            st.markdown("### 📊 Live Analytics")
            stats_placeholder = st.empty()
            
            # Performance metrics
            fps_placeholder = st.empty()
            detection_count_placeholder = st.empty()
    
    # Real-time detection loop
    detection_history = []
    frame_count = 0
    start_time = time.time()
    
    try:
        while st.session_state.get('camera_active', False):
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process frame
            annotated_frame, detections = process_image(
                frame, model, confidence_threshold
            )
            detection_history.extend(detections)
            
            # Calculate FPS
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                elapsed = current_time - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                fps_placeholder.markdown(f'''
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>📈 FPS:</strong> {current_fps:.1f}
                    </div>
                ''', unsafe_allow_html=True)
                
                detection_count_placeholder.markdown(f'''
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <strong>🎯 Total Detections:</strong> {len(detection_history)}
                    </div>
                ''', unsafe_allow_html=True)
            
            # Display frame
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb, 
                channels="RGB", 
                use_container_width=True,
                caption=f"Live Feed - Frame {frame_count}"
            )
            
            # Display real-time stats
            if detections:
                with stats_placeholder:
                    display_realtime_stats(detections)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.03)  # ~33 FPS max
            
    except Exception as e:
        st.error(f"❌ Error during real-time detection: {str(e)}")
    finally:
        cap.release()
        st.session_state.camera_active = False
        
        if detection_history:
            st.markdown("### 📊 Session Summary")
            display_enhanced_detection_stats(detection_history, time.time() - start_time)

if __name__ == "__main__":
    main()