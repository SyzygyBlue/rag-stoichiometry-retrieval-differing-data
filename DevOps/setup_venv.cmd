@echo off
setlocal

REM Project root is parent of this DevOps folder
set ROOT=%~dp0..
set VENV_DIR=%ROOT%\RAG Files\.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set REQS=%ROOT%\RAG Files\requirements.txt

if not exist "%VENV_DIR%" (
  echo [info] Creating virtual environment at "%VENV_DIR%" ...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [error] Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo [info] Using existing virtual environment at "%VENV_DIR%".
)

if not exist "%VENV_PY%" (
  echo [error] Expected venv Python at "%VENV_PY%" not found.
  exit /b 1
)

if not exist "%REQS%" (
  echo [warn] requirements.txt not found at "%REQS%". Installing minimal packages directly.
  "%VENV_PY%" -m pip install -U pip
  "%VENV_PY%" -m pip install -U sentence-transformers numpy requests
) else (
  echo [info] Installing requirements from "%REQS%" ...
  "%VENV_PY%" -m pip install -U pip
  "%VENV_PY%" -m pip install -r "%REQS%"
)

echo [done] Virtual environment ready: %VENV_PY%
exit /b 0
