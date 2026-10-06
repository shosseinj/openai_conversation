@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" goto existing
where py >nul 2>&1
if %ERRORLEVEL% equ 0 goto launcher
where python >nul 2>&1
if %ERRORLEVEL% equ 0 goto python
 echo Install Python 3.12 first, then double-click run.bat again.
pause
exit /b 1
:existing
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
goto finished
:launcher
py -3 "%~dp0run.py" %*
goto finished
:python
python "%~dp0run.py" %*
:finished
set "runnerExitCode=%ERRORLEVEL%"
if not "%runnerExitCode%"=="0" pause
exit /b %runnerExitCode%
