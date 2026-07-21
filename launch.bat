@echo off
setlocal enabledelayedexpansion
title ModelProject - Launch Pad
cd /d "%~dp0"

:: --- Colour helpers (Windows 10+ VT processing) ---
for /f "tokens=2 delims=:" %%a in ('chcp') do set "_cp=%%a"
set "_cp=%_cp: =%"
if "%_cp%"=="65001" ( set "G=[32m" & set "Y=[33m" & set "C=[36m" & set "R=[31m" & set "B=[1m" & set "D=[0m" )

:: --- Python discovery ---
:find_python
set "PY=python"
where python >nul 2>&1 || (
  where python3 >nul 2>&1 && set "PY=python3" || (
    echo %R%[FAIL]%D% Python not found on PATH. Install Python 3.14+ and try again.
    pause & exit /b 1
  )
)
call "%PY%" --version 2>&1 | findstr /R "3\.1[4-9] 3\.[2-9][0-9]" >nul
if errorlevel 1 (
  echo %Y%[WARN]%D% Expected Python ^>=3.14, got:
  "%PY%" --version
)

:: --- Virtual env auto-activate ---
if not defined VIRTUAL_ENV (
  for %%d in (.venv venv) do if exist "%%d\Scripts\activate.bat" (
    echo %C%[info]%D% Activating %%d ...
    call "%%d\Scripts\activate.bat"
    goto :env_done
  )
)
:env_done

:: --- Quick dependency check (run once, markers in .opencode/) ---
if not exist ".opencode\.deps_ok" (
  call "%PY%" -c "import fastapi, uvicorn, torch, pandas, pyarrow, yaml, httpx, numba, pytest" 2>nul
  if errorlevel 1 (
    echo.
    echo %Y%First launch - installing core dependencies...%D%
    call "%PY%" -m pip install --quiet -e . 2>nul
    if errorlevel 1 (
      call "%PY%" -m pip install --quiet fastapi uvicorn torch pandas pyarrow pyyaml httpx numba pytest 2>nul
    )
    call "%PY%" -c "import fastapi, uvicorn, torch, pandas, pyarrow, yaml, httpx, numba, pytest" 2>nul
    if NOT errorlevel 1 (
      mkdir ".opencode" 2>nul
      type nul > ".opencode\.deps_ok"
    )
  ) else (
    mkdir ".opencode" 2>nul
    type nul > ".opencode\.deps_ok"
  )
)

:: --- Handle --help / -h ---
if /i "%1"=="--help" goto :help
if /i "%1"=="-h" goto :help

:: --- Port detection ---
set "PORT=8000"
:: Check %1 first, then %2 (for "launch.bat serve 5000" style)
echo "%1" | >nul findstr /R "^\"[0-9][0-9]*\"" && set "PORT=%1"
echo "%2" | >nul findstr /R "^\"[0-9][0-9]*\"" && set "PORT=%2"

:: --- Direct mode: skip menu if arg is a known command ---
set "CMD="
set "CMD_ARGS="
if /i "%1"=="serve" set "CMD=serve"&goto strip_first
if /i "%1"=="nobrowser" set "CMD=serve_nobrowser"&goto strip_first
if /i "%1"=="compare" set "CMD=compare"&goto strip_first
if /i "%1"=="baseline" set "CMD=baseline"&goto strip_first
if /i "%1"=="test" set "CMD=tests"&goto strip_first
if /i "%1"=="tests" set "CMD=tests"&goto strip_first
if /i "%1"=="install" set "CMD=installdeps"&goto strip_first
if /i "%1"=="deps" set "CMD=installdeps"&goto strip_first
goto menu

:strip_first
shift
:arg_loop
if "%1"=="" goto :exec_cmd
set "CMD_ARGS=!CMD_ARGS! %1"
shift
goto arg_loop
:: NOTE: cmd splits batch args on = , ; - if you write --maxfail=1 it becomes --maxfail 1
:: (most tools accept both forms). & is not allowed in passthrough args.
:exec_cmd
goto %CMD%

:: =====================================================================
::  Menu
:: =====================================================================
:menu
cls
echo ============================================
echo  %B%ModelProject%D% - Intraday Backtesting
echo ============================================
echo.
echo  %C%[1]%D%  Start UI server  (port %PORT%)
echo  %C%[2]%D%  Start UI (no browser)
echo  %C%[3]%D%  Compare training runs
echo  %C%[4]%D%  Recompute baseline
echo  %C%[5]%D%  Run tests
echo  %C%[6]%D%  Install / update deps
echo  %C%[Q]%D%  Quit
echo.

:: Allow both keyboard choice and direct arg
choice /c 123456q /n /m "Select option (1-6, Q): "

:: choice sets errorlevel to the index (1-based)
set "_opt=%errorlevel%"
if "%_opt%"=="1" goto serve
if "%_opt%"=="2" goto serve_nobrowser
if "%_opt%"=="3" goto compare
if "%_opt%"=="4" goto baseline
if "%_opt%"=="5" goto tests
if "%_opt%"=="6" goto installdeps
if "%_opt%"=="7" exit /b
exit /b

:: =====================================================================
::  Serve (with browser)
:: =====================================================================
:serve
echo.
echo %C%[info]%D% Starting server at %G%http://127.0.0.1:%PORT%%D%
echo.
:: Launch server in a new window so we can poll before opening browser
:: NOTE: no "call" here - it breaks cmd /c quote-stripping. Bare %PY% is fine:
:: if python is a .bat shim, control-transfer is harmless in this dedicated window.
start "ModelProject Server" cmd /c ""%PY%" -m uvicorn ui.backend.main:app --host 127.0.0.1 --port %PORT% --reload & if errorlevel 1 pause"

:: Poll until server responds (up to ~2 min: first /series hit loads data + precomputes inference)
echo %Y%Waiting for server to start (up to ~2 min while inference precomputes)...%D%
set "ready="
for /l %%i in (1,1,90) do (
  >nul 2>&1 timeout /t 2 /nobreak
  <nul set /p "=."
  >nul 2>&1 curl -s http://127.0.0.1:%PORT%/series?symbol=SOLUSDT --connect-timeout 1 --max-time 120 && set "ready=1" && goto :browser_open
)
echo.
if not defined ready (
  echo %R%[WARN]%D% Server did not respond in time.
  echo %Y%Run 'launch.bat nobrowser' to see the server error in this window.%D%
  echo.
  pause
  exit /b 1
)
:browser_open
echo.
start http://127.0.0.1:%PORT%
echo %G%Server is live at http://127.0.0.1:%PORT%%D%
echo %C%Press Ctrl+C in the server window to stop.%D%
echo.
exit /b

:: =====================================================================
::  Serve (no browser)
:: =====================================================================
:serve_nobrowser
echo.
echo %C%[info]%D% Starting server at %G%http://127.0.0.1:%PORT%%D%
echo %C%Open http://127.0.0.1:%PORT% in your browser.%D%
echo.
call "%PY%" -m uvicorn ui.backend.main:app --host 127.0.0.1 --port %PORT% --reload
if errorlevel 1 (
  echo %R%[FAIL]%D% Server exited with code !errorlevel!. Make sure Python 3.14+ and dependencies are installed.
  pause
)
exit /b

:: =====================================================================
::  Compare runs
:: =====================================================================
:compare
echo.
call "%PY%" -m model.runs.compare %CMD_ARGS%
if errorlevel 1 echo %R%No runs found or error occurred.%D%
echo.
pause
exit /b

:: =====================================================================
::  Recompute baseline
:: =====================================================================
:baseline
echo.
call "%PY%" -m model.baselines.persistence_model
echo.
pause
exit /b

:: =====================================================================
::  Run tests
:: =====================================================================
:tests
echo.
call "%PY%" -m pytest -v %CMD_ARGS%
echo.
pause
exit /b

:: =====================================================================
::  Install / update deps
:: =====================================================================
:installdeps
echo.
echo Installing core dependencies...
call "%PY%" -m pip install --upgrade pip
call "%PY%" -m pip install -e . 2>nul
if errorlevel 1 (
  call "%PY%" -m pip install fastapi uvicorn torch pandas pyarrow pyyaml httpx numba pytest
)
del ".opencode\.deps_ok" 2>nul
echo %G%Dependencies installed.%D%
echo.
pause
goto menu

:: =====================================================================
::  Help
:: =====================================================================
:help
echo.
echo %B%USAGE:%D%
echo     launch.bat [port^|command]
echo.
echo %B%COMMANDS:%D%
echo   launch.bat serve        Start UI server + open browser
echo   launch.bat nobrowser    Start UI server only
echo   launch.bat compare      Compare training runs
echo   launch.bat baseline     Recompute persistence baseline
echo   launch.bat test(s)      Run test suite
echo   launch.bat install      Install / update dependencies
echo.
echo %B%PORT:%D%
echo   launch.bat 8080         Open menu with port 8080 pre-selected
echo.
echo %B%EXAMPLES:%D%
echo   launch.bat              Interactive menu
echo   launch.bat 3000         Menu on port 3000  (then pick option 1)
echo   launch.bat serve 5000   Start UI + browser on port 5000
echo   launch.bat test -x      Run tests with -x flag
echo   launch.bat compare --sort loss
echo.
pause
exit /b
