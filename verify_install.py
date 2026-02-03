"""
HOVES - Verify Installation Script
Run this to check if everything is set up correctly before running the app
"""

import sys
import importlib
from pathlib import Path

def check_python():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 8):
        print(f"[ERROR] Python 3.8+ required (found {version})")
        return False
    print(f"[OK] Python {version}")
    return True

def check_packages():
    packages = {
        'streamlit': 'Streamlit',
        'mediapipe': 'MediaPipe',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'google.generativeai': 'Google Generative AI',
        'dotenv': 'Python-dotenv',
    }
    
    all_ok = True
    for pkg, name in packages.items():
        try:
            importlib.import_module(pkg)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[ERROR] {name} - Run: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def check_files():
    files = [
        'app.py',
        'requirements.txt',
        'lib/analysis_engine.py',
    ]
    
    all_ok = True
    for f in files:
        if Path(f).exists():
            print(f"[OK] {f}")
        else:
            print(f"[ERROR] {f} missing")
            all_ok = False
    
    return all_ok

def check_env():
    if Path('.env').exists():
        print("[OK] .env file found")
        return True
    elif Path('.env.local').exists():
        print("[OK] .env.local file found")
        return True
    else:
        print("[WARN] .env file not found (chat features will be disabled)")
        print("   Create .env with: GEMINI_API_KEY=your_key_here")
        return True  # Not critical

def main():
    print("\n" + "="*50)
    print("HOVES - Installation Verification")
    print("="*50 + "\n")
    
    checks = [
        ("Python Version", check_python),
        ("Required Packages", check_packages),
        ("Project Files", check_files),
        ("Environment Variables", check_env),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 30)
        results.append(check_func())
    
    print("\n" + "="*50)
    if all(results):
        print("[OK] All checks passed! Ready to run:")
        print("   streamlit run app.py")
    else:
        print("[ERROR] Some checks failed. Please fix above issues.")
    print("="*50 + "\n")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
