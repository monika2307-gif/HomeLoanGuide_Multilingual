@echo off
title LoanGuide AI - Multilingual Assistant
echo ========================================================
echo        Starting LoanGuide AI (Streamlit Web App)
echo ========================================================
echo.

IF EXIST "C:\Users\%USERNAME%\anaconda3\Scripts\streamlit.exe" (
    echo Using Anaconda Python environment...
    "C:\Users\%USERNAME%\anaconda3\Scripts\streamlit.exe" run app.py
) ELSE IF EXIST "C:\Users\%USERNAME%\anaconda3\python.exe" (
    echo Using Anaconda Python executable...
    "C:\Users\%USERNAME%\anaconda3\python.exe" -m streamlit run app.py
) ELSE (
    echo Using default system Python/Streamlit...
    streamlit run app.py
)

pause
