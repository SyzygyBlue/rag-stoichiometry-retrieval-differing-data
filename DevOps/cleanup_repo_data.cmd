@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Cleanup non-program data from Git repo and update tree
REM - Deletes the local "Beyond the Neon Veil" folder
REM - Untracks scenes (*.json with "-scene"), character folder, and knowledge JSONL from Git
REM - Pushes changes to origin (optionally override remote URL as arg1)

REM 1) Verify git
where git >nul 2>nul
if errorlevel 1 (
  echo [error] git is not available on PATH. Please install Git for Windows.
  exit /b 1
)

REM 2) Ensure repo exists
if not exist .git (
  echo [fatal] This directory is not a git repository. Initialize it first.
  exit /b 2
)

REM 3) Delete local folder: Beyond the Neon Veil
if exist "Beyond the Neon Veil" (
  echo [rm] Deleting local folder: Beyond the Neon Veil
  rmdir /s /q "Beyond the Neon Veil"
) else (
  echo [rm] Beyond the Neon Veil not present locally (ok)
)

REM 4) Untrack data from Git index (keep files locally except the folder we removed)
set ERR=0

REM Untrack character folder
if exist "RAG Files\character" (
  git rm -r --cached -- "RAG Files/character" || set ERR=1
  echo [git] untracked RAG Files/character from index
) else (
  echo [git] character folder not found (ok)
)

REM Untrack knowledgeFile folder
if exist "RAG Files\knowledgeFile" (
  git rm -r --cached -- "RAG Files/knowledgeFile" || set ERR=1
  echo [git] untracked RAG Files/knowledgeFile from index
) else (
  echo [git] knowledgeFile folder not found (ok)
)

REM Untrack knowledge_qa.json if tracked
if exist "RAG Files\knowledge_qa.json" (
  git rm --cached -- "RAG Files/knowledge_qa.json" || set ERR=1
  echo [git] untracked RAG Files/knowledge_qa.json from index
) else (
  echo [git] knowledge_qa.json not found (ok)
)

REM Untrack scene JSONs that are currently tracked
for /f "usebackq tokens=*" %%F in (`git ls-files "RAG Files/*-scene*.json"`) do (
  git rm --cached -- "%%F" || set ERR=1
  echo [git] untracked %%F
)

REM Stage .gitignore and any deletions/modifications
git add -A || set ERR=1

REM 5) Commit
if %ERR% NEQ 0 (
  echo [warn] Some untrack operations returned errors; proceeding to commit what is staged.
)

git diff --cached --quiet
if errorlevel 1 (
  git commit -m "chore(cleanup): remove non-program data (scenes, character, knowledge) from repo; delete 'Beyond the Neon Veil'; update .gitignore" || goto :fail
  echo [git] committed cleanup
) else (
  echo [git] nothing to commit (cleanup may already be applied)
)

REM 6) Push (override remote URL if provided)
if not "%~1"=="" (
  git remote set-url origin "%~1"
)

git push -u origin main || goto :fail

echo ---
git remote -v
echo ---
git log -n 1 --oneline
echo ---
git status -s

REM Self-delete to keep repo clean
set "SELF=%~f0"
echo [cleanup] completed. Deleting %SELF% ...
(del /f /q "%SELF%") >nul 2>nul
exit /b 0

:fail
echo [fatal] Cleanup failed. Please review the messages above.
exit /b 2
