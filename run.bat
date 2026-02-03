@echo off
echo Starting HOVES Biometric Analysis Suite...
echo.

if exist hoves_env\Scripts\activate.bat (
    call hoves_env\Scripts\activate.bat
    streamlit run app.py
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    streamlit run app.py
) else (
    echo [WARNING] No virtual environment found. Attempting to run with global streamlit...
    streamlit run app.py
)

pause
