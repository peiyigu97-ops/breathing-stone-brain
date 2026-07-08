@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"

echo.
echo ================================================
echo   CHIMERA - Build EXE
echo ================================================
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause & exit /b 1
)

echo [1/5] Checking PyInstaller...
python -m PyInstaller --version > nul 2>&1
if errorlevel 1 (
    python -m pip install pyinstaller --quiet
)
echo     OK

echo [2/5] Checking connectome...
if not exist "chimera\connectome\real_weight_matrix.npy" (
    python chimera_load_connectome.py
    if errorlevel 1 ( echo [ERROR] connectome failed & pause & exit /b 1 )
)
echo     OK

echo [3/5] Checking LLM...
if exist "chimera\models\Qwen2.5-0.5B-Instruct.Q4_K_M.gguf" (
    echo     Qwen included
) else (
    echo     Qwen not found - rule-based mode
)

echo [4/5] Cleaning old build...
if exist "dist\CHIMERA"  rmdir /s /q "dist\CHIMERA"
if exist "build\CHIMERA" rmdir /s /q "build\CHIMERA"

echo [5/5] Building... (5-10 min)
echo.
python -m PyInstaller chimera.spec --noconfirm --clean
if errorlevel 1 (
    echo [ERROR] Build failed
    echo   - llama-cpp missing: pip install llama-cpp-python
    echo   - mujoco missing:    pip install mujoco
    pause & exit /b 1
)

if not exist "dist\CHIMERA\CHIMERA.exe" (
    echo [ERROR] CHIMERA.exe not found
    pause & exit /b 1
)

echo.
echo ================================================
echo   Build Success!
echo   dist\CHIMERA\CHIMERA.exe
echo   Distribute: ZIP dist\CHIMERA\ folder
echo ================================================
echo.

set /p RUN="Run now? (y/n): "
if /i "%RUN%"=="y" cd dist\CHIMERA && CHIMERA.exe

endlocal
pause
