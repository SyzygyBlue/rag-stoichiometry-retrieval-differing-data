@echo off
setlocal enabledelayedexpansion

REM Resolve project root as parent of this DevOps folder
set ROOT=%~dp0..
set VENV_PY="%ROOT%\RAG Files\.venv\Scripts\python.exe"
set RAG_QUERY="%ROOT%\RAG Files\scripts\rag_query.py"

if not exist %VENV_PY% (
  echo [error] Virtualenv Python not found at %VENV_PY%
  echo Create a venv at "RAG Files\.venv" and install requirements, e.g.:
  echo   python -m venv "RAG Files\.venv"
  echo   "RAG Files\.venv\Scripts\python.exe" -m pip install -U sentence-transformers numpy requests
  exit /b 1
)

if not exist %RAG_QUERY% (
  echo [error] rag_query.py not found at %RAG_QUERY%
  exit /b 1
)

%VENV_PY% %RAG_QUERY% %*
exit /b %errorlevel%
