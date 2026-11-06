@echo off
cd /d "%~dp0"
echo ======================================================================
echo  Uploading Case Study 3: Heart Disease Prediction (XAI) to GitHub
echo ======================================================================
echo.

set "GIT_PATH=C:\Users\kanav\AppData\Local\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd\git.exe"

"%GIT_PATH%" push -u origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Case Study 3 uploaded to GitHub successfully!
) else (
    echo [NOTICE] If GitHub requested authentication, you can also paste your
    echo          GitHub Personal Access Token (PAT) when prompted.
)
echo.
pause
