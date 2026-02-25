@echo off
TITLE GVM Setup Wizard
echo ===================================================
echo     GVM (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. 
    echo Please run Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

:: 1. Install Requirements
echo [1/2] Installing GVM specific dependencies...
call venv\Scripts\activate.bat
if exist "gvm_core\requirements.txt" (
    pip install -r gvm_core\requirements.txt
) else (
    echo Using main project dependencies for GVM...
)

:: 2. Download Weights
echo.
echo [2/2] Downloading GVM Model Weights (WARNING: Massive 80GB+ Download)...
if not exist "gvm_core\weights" mkdir "gvm_core\weights"

REM TODO: Add the actual download links for the GVM Stable Video Diffusion weights
echo NOTE: You still need to insert the weight download links into this batch script!
echo Downloading GVM weights...
REM curl.exe -L -o "gvm_core\weights\gvm_weight_1.safetensors" "LINK_HERE"

echo.
echo ===================================================
echo   GVM Setup Complete! 
echo ===================================================
pause
