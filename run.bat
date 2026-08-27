@echo off
setlocal EnableExtensions EnableDelayedExpansion
title CamToParkingSlot

REM ===========================================================================
REM  One file, one double-click, working product.
REM
REM  Named run.bat rather than start.bat because "start" is a cmd builtin, and a
REM  file that shadows a builtin is a trap for whoever types it next.
REM
REM  Written as .bat rather than PowerShell on purpose: a .bat runs the same
REM  whether it is double-clicked in Explorer, typed in cmd, called from
REM  PowerShell, or launched from Git Bash, and it never trips over the
REM  execution policy that blocks an unsigned .ps1 on a default Windows box.
REM
REM  It is safe to run twice. Every step checks whether it is already done, so
REM  a second run starts in seconds instead of re-downloading the Netherlands.
REM ===========================================================================

cd /d "%~dp0"

echo.
echo   CamToParkingSlot
echo   ================
echo.

REM --------------------------------------------------------------- prerequisites
set "MISSING="
where git >nul 2>&1 || set "MISSING=!MISSING! git"
where node >nul 2>&1 || set "MISSING=!MISSING! node"

if defined MISSING (
    echo   [X] Missing from PATH:!MISSING!
    echo.
    echo       git   https://git-scm.com/download/win
    echo       node  https://nodejs.org  ^(version 20 or newer^)
    echo.
    echo   Install what is listed, open a new terminal, and run this file again.
    goto :halt
)

REM uv manages Python and every dependency, so it is the only other thing needed.
where uv >nul 2>&1
if errorlevel 1 (
    echo   [ ] uv not found. Installing it, this takes about a minute...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo   [X] uv did not install. Install it by hand and run this again:
        echo       powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
        goto :halt
    )
    echo   [OK] uv installed
) else (
    echo   [OK] uv
)

echo   [OK] git, node

REM --------------------------------------------------------------- python side
REM --all-extras rather than a list of three. Naming a subset does not add to the
REM environment, it defines it, so uv removes everything outside the list. An earlier
REM version named three extras here and quietly uninstalled the plotting stack.
echo   [ ] Python packages...
call uv sync --all-extras >nul 2>&1
if errorlevel 1 (
    echo   [X] uv sync failed. Run it yourself to see why:
    echo       uv sync --all-extras
    goto :halt
)
echo   [OK] Python packages

REM --------------------------------------------------------------- web side
if not exist "web\node_modules" (
    echo   [ ] Web packages, first run only, a minute or two...
    pushd web
    call npm install >nul 2>&1
    if errorlevel 1 (
        echo   [X] npm install failed. Run it in the web folder to see why.
        popd
        goto :halt
    )
    popd
)
echo   [OK] Web packages

REM --------------------------------------------------------------- open data
REM The database is the slow part: parking bays, car parks, the road graph and
REM the points of interest, straight from the Dutch open data services. Ten to
REM twenty minutes once, then never again unless you delete it.
if not exist "data\parkfit.db" (
    echo.
    echo   [ ] No database yet. Pulling Dutch open data.
    echo       This runs once and takes 10 to 20 minutes. Leave it alone.
    echo.
    call uv run pf ingest all
    if errorlevel 1 (
        echo   [X] Ingest failed. You can retry with: uv run pf ingest all
        goto :halt
    )
)
echo   [OK] Open data

REM --------------------------------------------------------------- servers
REM Two windows rather than two background jobs, so their logs stay visible and
REM closing them is how you stop the product. Nothing is left running after.
echo   [ ] Starting API on 127.0.0.1:8000
start "CamToParkingSlot API" cmd /k ^
    "cd /d "%~dp0" && uv run uvicorn parkfit.api.app:app --host 127.0.0.1 --port 8000"

echo   [ ] Starting web on 127.0.0.1:5173
start "CamToParkingSlot Web" cmd /k ^
    "cd /d "%~dp0web" && npm run dev"

REM The API loads a 188,715-node road graph before it answers, so opening the
REM browser immediately shows a page that cannot search yet.
echo   [ ] Waiting for the API to come up...
set /a TRIES=0
:waitloop
set /a TRIES+=1
powershell -NoProfile -Command ^
    "try { (Invoke-WebRequest -Uri http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 3).StatusCode } catch { 0 }" ^
    | findstr /c:"200" >nul 2>&1
if not errorlevel 1 goto :ready
if !TRIES! GEQ 40 (
    echo   [!] API is slow to start. Opening the page anyway; check the API window.
    goto :ready
)
REM ping rather than timeout: timeout reads stdin and dies with "Input
REM redirection is not supported" the moment anyone pipes this to a log.
ping -n 4 127.0.0.1 >nul
goto :waitloop

:ready
echo   [OK] Running
echo.
start "" "http://127.0.0.1:5173"

echo   ---------------------------------------------------------------
echo    Open at   http://127.0.0.1:5173
echo    API docs  http://127.0.0.1:8000/docs
echo.
echo    To see a car checked against a bay:
echo      1. Press "Vehicles", top right
echo      2. Register with any email and password, it is your machine
echo      3. Add a car. "uv run pf cars" lists fourteen real ones
echo      4. Search, then pick it in the Vehicle dropdown
echo.
echo    Camera markers on the map open the live street view.
echo.
echo    Stop it by closing the two windows this opened.
echo   ---------------------------------------------------------------
echo.
echo   This window can be closed.
ping -n 16 127.0.0.1 >nul
exit /b 0

:halt
echo.
pause
exit /b 1
