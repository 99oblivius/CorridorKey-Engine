@echo off
TITLE GVM Setup Wizard
echo ===================================================
echo     GVM (AlphaHint Generator) - Auto-Installer
echo ===================================================
echo.

:: Check that uv sync has been run (the .venv directory should exist)
if not exist ".venv" (
    echo [ERROR] Project environment not found.
    echo Please run tools\Install_CorridorKey_Windows.bat first!
    pause
    exit /b
)

:: 1. Download Weights (all Python deps are already installed by uv sync)
echo [1/1] Downloading GVM Model Weights (WARNING: Massive 80GB+ Download)...
if not exist "alpha_generators\gvm\weights" mkdir "alpha_generators\gvm\weights"

echo Downloading GVM weights from HuggingFace...
uv run hf download geyongtao/gvm --local-dir alpha_generators\gvm\weights

echo.
echo ===================================================
echo   GVM Setup Complete!
echo ===================================================
pause
