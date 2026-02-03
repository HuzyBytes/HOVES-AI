# HOVES - Biometric Analysis Suite (Python Edition)

Advanced facial anthropometry and aesthetic optimization using medical-grade computer vision.

## Overview

HOVES is a sophisticated facial biometric analysis system built entirely in Python that combines:
- **MediaPipe Face Mesh** for real-time facial landmark detection
- **Advanced metrics calculation** (symmetry, golden ratio, canthal tilt, etc.)
- **AI-powered analysis** via Gemini API
- **Beautiful clinical UI** with Streamlit

## Technology Stack

### Complete Python Stack
- **Streamlit** - Modern web UI framework (100% Python)
- **MediaPipe** - Face mesh detection (468-point facial landmark detection)
- **OpenCV** - Image processing and face mesh visualization
- **NumPy** - Numerical calculations and metrics
- **Google Generative AI** - Gemini chat integration
- **Pillow** - Image manipulation

## Installation

### 1. Navigate to Project Directory

```bash
cd "f:\HUZY CODE STUFF !\Python\HOVES"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

Create a `.env` file (or use existing `.env.local`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://makersuite.google.com/app/apikey

## Running the Application

```bash
streamlit run app_streamlit.py
```

Streamlit will automatically open your browser to the local application (usually `http://localhost:8501`)

If it doesn't open automatically, navigate to:
```
http://localhost:8501
```

## Features

### 🎯 Biometric Scanner
- **Live Webcam Scanning**: Real-time face mesh detection
- **Image Upload**: Analyze photos
- **Face Mesh Overlay**: Golden wireframe visualization
- **Scan Animation**: Clinical scan line effect

### 📊 Assessment Sidebar
- **Composite Score**: Overall aesthetic rating (1-10)
- **Accordion Sections**:
  - I. Orbital Region (canthal tilt, eye spacing)
  - II. Midface & Nose (golden ratio, zygomatic width)
  - III. Lower Third & Jaw (mandibular angle, beard density)
  - IV. Skin & Texture (clarity, hair type)

### 💬 AI Chat (Aesthetica)
- **Gemini-Powered**: Intelligent analysis and recommendations
- **Context-Aware**: Uses your facial metrics
- **Clinical Tone**: Data-driven, objective feedback

### ✨ Enhancement Suite
- **Before/After Slider**: Interactive comparison
- **AI Simulation**: Visual enhancement preview
- **Processing Animation**: Professional UI

## Project Structure

```
HOVES/
├── app_streamlit.py           # Main Streamlit application (Python)
├── app.py                     # Original Flask app (deprecated)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this!)
├── .env.local                 # Local env variables
├── README.md                  # This file
├── api/
│   ├── __init__.py
│   ├── chat.py                # Gemini chat logic
│   └── analyze.py             # Face analysis (legacy, now in lib)
├── lib/
│   ├── __init__.py
│   └── analysis_engine.py     # Core metrics calculation
├── static/
│   └── css/
│       └── styles.css         # Streamlit custom CSS
└── templates/
    └── index.html             # Legacy HTML (not used by Streamlit)
```

## Usage

1. **Start the Streamlit app**: `streamlit run app_streamlit.py`
2. **Click "INITIATE SESSION"** on the landing page
3. **Choose scanner mode**:
   - **Live Camera**: Real-time webcam scanning
   - **Upload Photo**: Analyze an existing image
4. **Click "🔍 Analyze Face"** to process landmarks
5. **View metrics** in the Assessment Results panel
6. **Open Aesthetica AI Chat** for deeper analysis and recommendations

## Features

### 🎥 Biometric Scanner
- **Live Webcam Scanning**: Real-time face mesh detection via camera
- **Image Upload**: Analyze photos from disk
- **Face Mesh Visualization**: Golden wireframe overlay (468 facial landmarks)
- **Automatic Analysis**: One-click facial metrics calculation

### 📊 Assessment Results
Expandable sections showing:
- **Overall Score**: 1-10 aesthetic rating
- **Orbital Metrics**: Canthal tilt, eye spacing
- **Proportion Metrics**: Golden ratio, midface ratio, symmetry
- **Mandibular Metrics**: Jawline definition
- **Grooming Assessment**: Beard density, hair type
- **Skin Analysis**: Clarity score

### 🤖 Aesthetica AI Chat
- **Gemini-Powered Analysis**: Intelligent recommendations
- **Context-Aware**: Uses your facial metrics in responses
- **Clinical Tone**: Direct, data-driven feedback
- **Real-time Conversation**: Chat interface within the app

## Metrics Explained

### Symmetry (30% weight)
Measures left-right facial balance across key features (eyes, jaw, cheeks). Range: 0-100%

### Golden Ratio (40% weight)
Evaluates facial proportions against the ideal 1.618 ratio. Ideal value: 1.618

### Canthal Tilt
Angle of the eye's outer corner relative to inner corner
- Positive tilt (>2°) = "Hunter Eyes" aesthetic ideal
- Neutral tilt (0-2°) = Average
- Negative tilt (<0°) = Downturned eyes

### Mandibular Angle
Jawline definition angle. Ideal: ~120°

### Skin Clarity Score
Visual assessment of skin texture and quality (1-100)

## API Reference

### Core Module: `lib.analysis_engine`

#### `analyze_face(landmarks_data: List[Dict]) -> AnalysisMetrics`
Main analysis function that calculates all facial metrics from MediaPipe landmarks.

**Input:**
- `landmarks_data`: List of 468 landmark dictionaries with `x`, `y`, `z` coordinates (0-1 normalized)

**Returns:** `AnalysisMetrics` object with:
```python
{
    "basic": {
        "symmetry": float,        # 0-100%
        "skin_clarity": int,      # 1-100
        "golden_ratio": float,    # Ideal: 1.618
        "overall_score": float    # 1-10
    },
    "expert": {
        "canthal_tilt": float,    # degrees
        "intercanthal_distance": float,
        "mandibular_angle": float,
        "zygomatic_prominence": float,
        "midface_ratio": float,
        "facial_thirds": [float, float, float]
    },
    "grooming": {
        "forehead_height": float,
        "beard_density": int,     # 0-100
        "hair_volume": float,
        "hair_type": str
    }
}
```

## Troubleshooting

### Camera Permission Issues
- **Windows/Mac/Linux**: Check browser settings for camera access
- **HTTP vs HTTPS**: Streamlit uses localhost (HTTP) which is allowed for camera
- **Alternative**: Use "Upload Photo" mode

### MediaPipe Detection Failures
- Ensure good lighting
- Face should be 30-70% of frame width
- Face angles up to 45° are supported

### Gemini API Errors
- **"API Key not found"**: Create `.env` with `GEMINI_API_KEY=...`
- **"Rate limit exceeded"**: Wait 60 seconds before next request
- **"Model offline"**: Try later or check Gemini API status

### Streamlit Performance
- For slower systems, reduce webcam resolution
- Browser refresh (`Ctrl+R`) if UI becomes unresponsive
- Ensure `venv` is properly activated before running

## Credits

- **Original Stack**: Next.js/React/TypeScript with Flask backend
- **Current Stack**: 100% Python (Streamlit, MediaPipe, Gemini AI)
- **MediaPipe**: Google's advanced face mesh detection
- **Gemini API**: Google's generative AI for analysis
- **Streamlit**: Rapid Python app development framework

## License

Educational/Research Use Only

---

### ⚠️ Important Disclaimer

This application is for **educational and research purposes only**. Facial analysis scores are **algorithmic estimates** and should NOT be:
- Considered medical or professional assessments
- Used for discrimination
- Treated as objective truth about appearance or worth

Facial aesthetics are **subjective** and culturally dependent. The "beauty metrics" in this tool are based on specific historical ideals and should be understood as such.

---

**Version**: 2.1 (Python/Streamlit Edition)
**Last Updated**: 2026-01-25
