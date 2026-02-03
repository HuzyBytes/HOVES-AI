"""
HOVES - Biometric Analysis Suite (Streamlit Edition)
Advanced facial anthropometry and aesthetic optimization using medical-grade computer vision
Converted from Next.js/React to Pure Python
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

# Import MediaPipe Tasks API (v0.10+)
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image as MPImage

# Import custom modules
from lib.analysis_engine import analyze_face

# ==================== CONFIG ====================
load_dotenv()

# Configure page
st.set_page_config(
    page_title="HOVES - Biometric Analysis Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.error("⚠️ OpenAI API Key not found in .env file")

# ==================== STYLING ====================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    /* Main theme */
    :root {
        --primary: #000000;
        --accent: #D4AF37;
        --accent-muted: #C5A059;
        --background: #000000;
        --surface: #0A0A0A;
        --surface-light: #121212;
        --text: #FFFFFF;
        --text-muted: #888888;
        --border: #222222;
        --font-serif: 'Playfair Display', serif;
        --font-sans: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: var(--font-sans);
    }
    
    /* Header & Typography */
    h1, h2, h3 {
        font-family: var(--font-serif) !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }
    
    p, span, label, div {
        color: var(--text) !important;
        font-family: var(--font-sans);
    }

    /* Landing Page Hero */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 1rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-label {
        color: var(--accent) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3em !important;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    .hero-title {
        font-size: 5rem !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, #FFF 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: var(--text-muted) !important;
        font-size: 1.1rem !important;
        max-width: 600px;
        line-height: 1.6;
        margin-bottom: 3rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: transparent !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background-color: var(--accent) !important;
        color: black !important;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
    }

    /* Custom Cards/Sections */
    .clinical-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border);
    }
    
    .metric-label {
        color: var(--text-muted) !important;
        font-size: 0.9rem;
    }
    
    .metric-value {
        font-weight: 600;
        color: var(--accent) !important;
    }

    /* Score Display */
    .score-container {
        text-align: center;
        padding: 3rem 1rem;
        background: radial-gradient(circle at center, rgba(212, 175, 55, 0.05) 0%, transparent 70%);
    }
    
    .score-large {
        font-family: var(--font-serif);
        font-size: 6rem;
        line-height: 1;
        color: var(--accent) !important;
        margin: 1rem 0;
    }
    
    /* Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent !important;
        border-radius: 0 !important;
        padding: 10px 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button p {
        font-size: 0.9rem !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* MediaPipe specific */
    div[data-testid="stCameraInput"] {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Welcome to HOVES Biometric Suite. Upload scan data for clinical aesthetic evaluation."}
        ]
    
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'webcam'
    
    if 'landmarks' not in st.session_state:
        st.session_state.landmarks = None

init_session_state()

# ==================== MEDIAPIPE SETUP ====================
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_PATH = "face_landmarker.task"

def download_face_landmarker():
    if not os.path.exists(FACE_LANDMARKER_PATH):
        import urllib.request
        try:
            urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        except Exception as e:
            return False
    return True

face_landmarker = None
try:
    if download_face_landmarker():
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1
        )
        face_landmarker = FaceLandmarker.create_from_options(options)
except Exception as e:
    pass

# ==================== UTILITY FUNCTIONS ====================
def preprocess_image(image_rgb):
    if len(image_rgb.shape) == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    elif image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
    
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def extract_landmarks(image_rgb):
    if face_landmarker is None:
        return None, None
    
    try:
        image_processed = preprocess_image(image_rgb)
        if image_processed.dtype != np.uint8:
            image_processed = (image_processed * 255).astype(np.uint8)
        
        try:
            from mediapipe.tasks.python.vision.core.image import ImageFormat
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=image_processed)
        except:
            mp_image = MPImage(image_format=1, data=image_processed)
        
        detection_result = face_landmarker.detect(mp_image)
        
        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            landmarks = detection_result.face_landmarks[0]
            if len(landmarks) >= 468:
                landmarks_data = [{'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)} for lm in landmarks]
                return landmarks_data, landmarks
    except Exception as e:
        st.error(f"Detection Error: {e}")
    return None, None

def draw_face_mesh(image, landmarks):
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if landmarks:
        lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        for i, lm in enumerate(lm_list):
            x = int(lm.x * w)
            y = int(lm.y * h)
            x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            
            # Clinical Gold Points
            cv2.circle(image_rgb, (x, y), 1, (212, 175, 55), -1)
            # Subtle glow for major landmarks (eyes, lips, jaw)
            if i in [1, 33, 263, 61, 291, 152, 10]:
                cv2.circle(image_rgb, (x, y), 3, (212, 175, 55), 1)
    
    return PILImage.fromarray(image_rgb)

def analyze_landmarks(landmarks_data):
    try:
        metrics = analyze_face(landmarks_data)
        return metrics.to_dict()
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def chat_with_openai(messages, metrics=None):
    if not client:
        return "⚠️ OpenAI API Key not configured."
    
    try:
        system_prompt = """STRICT CLINICAL RATING persona (Qoves style). 
Be objective, data-driven, and slightly cold. Use medical terminology (e.g., 'bizygomatic width', 'canthal tilt', 'gonial angle'). 
Keep responses highly structured and professional."""
        
        context = ""
        if metrics:
            basic = metrics.get('basic', {})
            expert = metrics.get('expert', {})
            context = f"[METRICS: Score={basic.get('overall_score')}, Symmetry={basic.get('symmetry')}%, Tilt={expert.get('canthal_tilt')}deg]"
        
        openai_messages = [{"role": "system", "content": system_prompt + "\n" + context}]
        for msg in messages:
            openai_messages.append(msg)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=openai_messages,
            temperature=0.6,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==================== PAGE: LANDING ====================
def page_landing():
    """Qoves-inspired Dynamic Landing Page"""
    hero_img_path = Path("C:/Users/BattleStation/.gemini/antigravity/brain/1b934af0-553d-4b97-8927-afe51bf1ad3a/hoves_hero_scan_1769537834627.png")
    
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    
    # Hero Label
    st.markdown('<p class="hero-label">Advanced Biometric Intelligence</p>', unsafe_allow_html=True)
    
    # Hero Title
    st.markdown('<h1 class="hero-title">HOVES</h1>', unsafe_allow_html=True)
    
    # Hero Subtitle
    st.markdown('<p class="hero-subtitle">Medical-grade facial anthropometry and aesthetic optimization using state-of-the-art computer vision and proprietary analysis logic.</p>', unsafe_allow_html=True)
    
    # Hero Image (Centered)
    if hero_img_path.exists():
        st.image(str(hero_img_path), width=600)
    
    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    # CTA
    if st.button("INITIATE CLINICAL SCAN"):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 2rem; width: 100%; text-align: center; color: #444; font-size: 0.75rem; letter-spacing: 0.2em;">
        SYSTEM v3.0 // AES7-X PROTOCOL
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: DASHBOARD ====================
def page_dashboard():
    """Main dashboard with clinical presentation"""
    
    # Top Progress Bar (Simulated)
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, var(--accent) 30%, #111 100%); width: 100%;"></div>', unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<h2 style="margin: 2rem 0 0.5rem 0;">BIOMETRIC WORKSTATION</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666 !important; font-size: 0.8rem; letter-spacing: 0.1em; margin-bottom: 2rem;">STATUS: QUANTUM ANALYZER ACTIVE</p>', unsafe_allow_html=True)

    with col2:
        if st.button("← RETURN TO LANDING", key="back_btn"):
            st.session_state.page = 'landing'
            st.rerun()

    # Layout
    col_main, col_metrics = st.columns([3, 2], gap="large")
    
    with col_main:
        tabs = st.tabs(["LIVE SCANNER", "DATA UPLOAD"])
        
        with tabs[0]:
            pic = st.camera_input("Scanner Feed")
            if pic:
                img = PILImage.open(pic)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                l_data, l_raw = extract_landmarks(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                
                if l_data:
                    st.session_state.landmarks = l_data
                    st.session_state.captured_image = img_cv
                    st.image(draw_face_mesh(img_cv, l_raw), use_container_width=True)
                    if st.button("EXECUTE ANALYSIS"):
                        st.session_state.metrics = analyze_landmarks(l_data)
                        st.rerun()

        with tabs[1]:
            upld = st.file_uploader("Upload Profile Image", type=["jpg", "png"])
            if upld:
                img = PILImage.open(upld)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                l_data, l_raw = extract_landmarks(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                if l_data:
                    st.session_state.landmarks = l_data
                    st.session_state.captured_image = img_cv
                    st.image(draw_face_mesh(img_cv, l_raw), use_container_width=True)
                    if st.button("EXECUTE UPLOAD ANALYSIS"):
                        st.session_state.metrics = analyze_landmarks(l_data)
                        st.rerun()

    with col_metrics:
        if st.session_state.metrics:
            m = st.session_state.metrics
            b = m.get('basic', {})
            e = m.get('expert', {})
            i = m.get('insights', {})
            
            # Score
            st.markdown('<div class="score-container">', unsafe_allow_html=True)
            st.markdown('<p style="color: #888 !important; letter-spacing: 0.2em; font-size: 0.75rem;">AESTHETIC HARMONY</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-large">{b.get("overall_score"):.1f}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: var(--accent) !important; font-size: 0.8rem;">PERCENTILE: {int(b.get("overall_score")*10)}th</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Insights Tabs
            insight_tabs = st.tabs(["BIOMETRICS", "FUN FACTS", "RECOMMENDATIONS", "ROUTINE GUIDE"])
            
            with insight_tabs[0]:
                st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                stats = [
                    ("Facial Symmetry", f"{b.get('symmetry')}%"),
                    ("Canthal Tilt", f"{e.get('canthal_tilt')}°"),
                    ("Mandibular Angle", f"{e.get('mandibular_angle')}°"),
                    ("Midface Ratio", b.get('golden_ratio')),
                    ("Intercanthal Ratio", e.get('intercanthal_distance'))
                ]
                for lbl, val in stats:
                    st.markdown(f'<div class="metric-item"><span class="metric-label">{lbl}</span><span class="metric-value">{val}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with insight_tabs[1]:
                st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                for fact in i.get('fun_facts', []):
                    st.markdown(f'<p style="font-size: 0.9rem; margin-bottom: 1rem; color: #BBB !important;">• {fact}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with insight_tabs[2]:
                st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                for rec in i.get('recommendations', []):
                    st.markdown(f'<p style="font-size: 0.9rem; margin-bottom: 1rem; color: var(--accent) !important;">• {rec}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with insight_tabs[3]:
                st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                for step in i.get('routine_guide', []):
                    st.markdown(f'<p style="font-size: 0.9rem; margin-bottom: 1rem; border-left: 2px solid var(--accent); padding-left: 10px;">{step}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Chat
            st.markdown('<h3 style="margin-top: 2rem;">AESTHETICA AI</h3>', unsafe_allow_html=True)
            chat_box = st.container(height=300)
            with chat_box:
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            if q := st.chat_input("Query the analyst..."):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                with st.spinner("Processing..."):
                    resp = chat_with_openai(st.session_state.chat_messages, m)
                    st.session_state.chat_messages.append({"role": "assistant", "content": resp})
                st.rerun()
        else:
            st.markdown('<div style="padding: 5rem 2rem; text-align: center; border: 1px dashed #333; border-radius: 8px; color: #444 !important;">WAITING FOR BIOMETRIC DATA INPUT</div>', unsafe_allow_html=True)

# ==================== MAIN APP FLOW ====================
def main():
    if st.session_state.page == 'landing':
        page_landing()
    else:
        page_dashboard()

if __name__ == "__main__":
    main()
