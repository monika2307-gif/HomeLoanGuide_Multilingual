@echo off
title Push to GitHub - LoanGuide AI
cd /d "C:\Users\amoln\OneDrive\Documents\Loan_Guidelines(project)"
echo ======================================================================
echo           Pushing Multilingual LoanGuide AI to GitHub
echo ======================================================================
echo.
echo Target: https://github.com/monika2307-gif/HomeLoanGuide_Multilingual.git
echo Branch: main
echo.
echo If a browser window appears asking you to sign in or authorize GitHub,
echo please click 'Sign in with your browser' / 'Authorize'.
echo.
"C:\Users\amoln\anaconda3\Library\cmd\git.exe" push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ======================================================================
    echo   🎉 SUCCESS! All files have been pushed to GitHub!
    echo ======================================================================
) else (
    echo.
    echo ⚠️ Push did not complete. If you were prompted to authenticate,
    echo    please verify your login or generate a token at https://github.com/settings/tokens
)
echo.
pause
