@echo off
echo 🚀 Starting Time Series Analysis Chatbot...
echo.
echo This will open a web browser with the interactive chatbot interface.
echo If it doesn't work, try: streamlit run streamlit_app.py
echo.
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run streamlit_app.py
pause
