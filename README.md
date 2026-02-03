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

## Implementation & Troubleshooting

### API Requirements:
To enable AI analysis, generate a key via the Google AI Studio and add it to the .env file: GEMINI_API_KEY=your_key_here
Detection Optimization:
Lighting: Ensure front-facing, even lighting to prevent landmark jitter.
Positioning: Face should occupy 30-70% of the frame for optimal MediaPipe accuracy.
Connectivity: Check the Google Cloud Status if the Aesthetica chat fails to respond.

## ⚖️ Ethical Disclaimer

Educational and Research Use Only. Facial metrics are algorithmic estimates based on specific historical and cultural datasets. These scores are subjective and should not be used as professional medical advice or for discriminatory purposes.



