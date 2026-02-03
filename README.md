<h1>HOVES: Biometric Analysis Suite (v2.1)<h1>
  
HOVES is a medical-grade facial anthropometry system that leverages computer vision and generative AI to provide data-driven aesthetic assessments. Built on a 100% Python stack, it transforms 468-point facial landmarks into clinical metrics and actionable grooming insights.

🛠 Technical Architecture
Component	Technology	Role
Interface	Streamlit	Modern, reactive Python web UI
Detection	MediaPipe Face Mesh	468-point real-time landmark tracking
Logic	NumPy / OpenCV	Matrix-based geometric calculations & image processing
Intelligence	Google Gemini API	Context-aware clinical analysis & chat (Aesthetica)

🚀 Key Features
1. Biometric Scanner
Dual Mode Acquisition: Supports real-time webcam streaming and high-resolution photo uploads.
Golden Wireframe Overlay: Visualizes facial topology using a real-time mesh projection.
Clinical UI: Features professional scan-line animations and a processing "Enhancement Suite" with before/after comparisons.
2. Advanced Metrics Engine
The system calculates weighted aesthetic scores based on the following biometric categories:
Symmetry (30% weight): Bilateral balance between eyes, cheeks, and jaw.
Golden Ratio (40% weight): Proportional alignment with the 1.618 ideal.
Orbital Region: Canthal tilt (Hunter eyes vs. downturned) and intercanthal distance.
Mandibular Region: Jawline definition and 120° mandibular angle assessment.
Dermatological Assessment: Computer-vision-based skin clarity and texture scoring.
3. Aesthetica AI Chat
A specialized Gemini-powered assistant that consumes biometric data to provide:
Contextual Feedback: Insights based on specific facial metrics.
Grooming Recommendations: Tailored advice for hair volume and beard density.
Objective Tone: Data-driven, clinical conversation style.

📂 System Structure
text
HOVES/
├── app_streamlit.py       # Primary Application Entry Point
├── lib/
│   └── analysis_engine.py # Core biometric logic & geometric math
├── api/
│   └── chat.py            # Gemini LLM integration & prompt engineering
├── static/
│   └── css/styles.css     # Custom clinical UI branding
└── .env                   # Configuration (API Keys & Local Vars)
Use code with caution.

⚙️ Implementation & Troubleshooting
API Requirements:
To enable AI analysis, generate a key via the Google AI Studio and add it to the .env file: GEMINI_API_KEY=your_key_here
Detection Optimization:
Lighting: Ensure front-facing, even lighting to prevent landmark jitter.
Positioning: Face should occupy 30-70% of the frame for optimal MediaPipe accuracy.
Connectivity: Check the Google Cloud Status if the Aesthetica chat fails to respond.

⚖️ Ethical Disclaimer
Educational and Research Use Only. Facial metrics are algorithmic estimates based on specific historical and cultural datasets. These scores are subjective and should not be used as professional medical advice or for discriminatory purposes.
