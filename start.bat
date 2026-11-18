@echo off
setlocal
rem Windows: double-click, or start.bat [--api-only ^| --ui-only] [--no-browser]
pushd "%~dp0" || exit /b 1

if exist ".venv\Scripts\python.exe" goto use_venv
where py >nul 2>nul
if errorlevel 1 goto try_python
py -3.12 -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul
if errorlevel 1 goto try_python
py -3.12 "scripts\launch.py" %*
goto finish

:use_venv
".venv\Scripts\python.exe" "scripts\launch.py" %*
goto finish

:try_python
where python >nul 2>nul
if errorlevel 1 goto missing_python
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 12))" >nul 2>nul
if errorlevel 1 goto missing_python
python "scripts\launch.py" %*
goto finish

:missing_python
echo PathSnap needs Python 3.12. Install it, then run start.bat again.
set "PATHSNAP_EXIT=1"
goto cleanup

:finish
set "PATHSNAP_EXIT=%ERRORLEVEL%"

:cleanup
popd
if "%PATHSNAP_EXIT%"=="0" exit /b 0
echo.
echo PathSnap could not start. See the error above.
if not "%PATHSNAP_NO_PAUSE%"=="1" pause
exit /b %PATHSNAP_EXIT%
