<#
.SYNOPSIS
    CamToParkingSlot task runner for Windows.

.DESCRIPTION
    One entry point for every routine operation. It locates the MSVC toolchain from
    Visual Studio Build Tools (including the CMake and Ninja that ship inside it), so
    the C++ tree builds on a stock Windows machine with no separate CMake install and
    no WSL detour.

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 build
    .\tasks.ps1 test
    .\tasks.ps1 ingest -Args "--city amsterdam"
    .\tasks.ps1 serve
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'build', 'rebuild', 'test', 'pytest', 'ctest', 'lint', 'format',
                 'ingest', 'serve', 'web', 'eval', 'cv', 'audit', 'clean', 'help')]
    [string]$Task = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$BuildDir = Join-Path $Root 'build'

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Toolchain discovery
# ---------------------------------------------------------------------------
function Get-VsInstall {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Install Visual Studio Build Tools 2022 with the C++ workload." }
    $path = & $vswhere -products * -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $path) {
        $path = & $vswhere -products * -latest -property installationPath
    }
    if (-not $path) { throw "No Visual Studio installation with the C++ toolset was found." }
    return $path.Trim()
}

function Get-CMakeExe {
    $c = Get-Command cmake -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $vs = Get-VsInstall
    $bundled = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path $bundled) { return $bundled }
    throw "cmake.exe not found on PATH nor inside $vs"
}

function Get-NinjaExe {
    $c = Get-Command ninja -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $vs = Get-VsInstall
    $bundled = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
    if (Test-Path $bundled) { return $bundled }
    return $null
}

function Get-CTestExe {
    $cmake = Get-CMakeExe
    $ctest = Join-Path (Split-Path $cmake -Parent) 'ctest.exe'
    if (Test-Path $ctest) { return $ctest }
    $c = Get-Command ctest -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    throw "ctest.exe not found next to $cmake"
}

# Import the MSVC environment into this PowerShell session by running vcvars64.bat
# in a child cmd and harvesting the resulting environment block. This is the
# supported way to get cl.exe, the linker and the Windows SDK onto PATH.
$script:MsvcEnvLoaded = $false
function Import-MsvcEnv {
    if ($script:MsvcEnvLoaded) { return }
    $vs = Get-VsInstall
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    Write-Ok "MSVC environment from $vs"
    $output = & "$env:ComSpec" /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $output) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            if ($name -notmatch '^(PROMPT|_)$') {
                Set-Item -Path "Env:$name" -Value $value -ErrorAction SilentlyContinue
            }
        }
    }
    $script:MsvcEnvLoaded = $true
}

function Get-Venv {
    $py = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { throw "Virtualenv missing. Run: .\tasks.ps1 setup" }
    return $py
}

# Windows PowerShell 5.1 turns any stderr output from a native executable into an
# ErrorRecord, which under $ErrorActionPreference = 'Stop' aborts on a mere warning.
# CMake writes warnings to stderr routinely, so native calls run with the preference
# relaxed and are judged purely on their exit code, which is the real signal.
function Invoke-Native([string]$exe, [string[]]$argv) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $exe @argv 2>&1 | ForEach-Object { "$_" }
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Invoke-Checked([string]$exe, [string[]]$argv, [string]$what) {
    Invoke-Native $exe $argv
    if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE" }
}

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
function Task-Setup {
    Write-Step 'Python environment'
    Push-Location $Root
    try {
        Invoke-Native 'uv' @('sync','--extra','dev','--extra','audit','--extra','ml')
        if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
        Invoke-Native 'uv' @('add','pybind11') | Out-Null
        Write-Ok 'dependencies installed'
    } finally { Pop-Location }

    Write-Step 'C++ toolchain'
    Import-MsvcEnv
    $cmake = Get-CMakeExe
    Write-Ok "cmake  $cmake"
    $ninja = Get-NinjaExe
    if ($ninja) { Write-Ok "ninja  $ninja" } else { Write-Warn2 'ninja not found, falling back to the MSBuild generator' }
    Task-Configure
    Write-Step 'Done. Next: .\tasks.ps1 build'
}

function Task-Configure {
    Import-MsvcEnv
    $cmake = Get-CMakeExe
    $ninja = Get-NinjaExe
    $py = Join-Path $Root '.venv\Scripts\python.exe'

    $argv = @('-S', $Root, '-B', $BuildDir, '-DCMAKE_BUILD_TYPE=Release')
    if ($ninja) {
        $argv += @('-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja")
    } else {
        $argv += @('-G', 'Visual Studio 17 2022', '-A', 'x64')
    }
    if (Test-Path $py) { $argv += "-DPython3_EXECUTABLE=$py" }

    Write-Step 'CMake configure'
    Invoke-Checked $cmake $argv 'cmake configure'
}

function Task-Build {
    if (-not (Test-Path (Join-Path $BuildDir 'CMakeCache.txt'))) { Task-Configure }
    Import-MsvcEnv
    $cmake = Get-CMakeExe
    Write-Step 'CMake build'
    Invoke-Checked $cmake @('--build', $BuildDir, '--config', 'Release', '--parallel') 'cmake build'
    Write-Ok 'C++ build complete'
}

function Task-Rebuild {
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    Task-Configure
    Task-Build
}

function Task-CTest {
    Task-Build
    Import-MsvcEnv
    $ctest = Get-CTestExe
    Write-Step 'C++ tests'
    Invoke-Native $ctest @('--test-dir', $BuildDir, '--output-on-failure', '-C', 'Release')
    if ($LASTEXITCODE -ne 0) { throw "ctest failed with exit code $LASTEXITCODE" }
    Write-Ok 'C++ tests passed'
}

function Task-Pytest {
    $py = Get-Venv
    Write-Step 'Python tests'
    $argv = @('-m', 'pytest', '-q')
    if ($Rest) { $argv += $Rest }
    Invoke-Native $py $argv
    if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }
    Write-Ok 'Python tests passed'
}

function Task-Test {
    Task-CTest
    Task-Pytest
}

function Task-Lint {
    $py = Get-Venv
    Write-Step 'ruff'
    Invoke-Native $py @('-m','ruff','check','src','tests')
    if ($LASTEXITCODE -ne 0) { throw 'ruff found issues' }
    # The same check CI runs. Without it, formatting drift passes locally and fails on
    # push, which is exactly what happened twice: the linter was clean and the formatter
    # was not, and nothing here would have caught it.
    Invoke-Native $py @('-m','ruff','format','--check','src','tests')
    if ($LASTEXITCODE -ne 0) { throw 'ruff format would reformat files; run .\tasks.ps1 fmt' }
    Write-Step 'mypy'
    Invoke-Native $py @('-m','mypy')
    Write-Ok 'lint clean'
}

function Task-Format {
    $py = Get-Venv
    Invoke-Native $py @('-m','ruff','format','src','tests')
    Invoke-Native $py @('-m','ruff','check','--fix','src','tests')
    Write-Ok 'formatted'
}

function Task-Pf([string[]]$pfArgs) {
    $py = Get-Venv
    Invoke-Native $py (@('-m','parkfit.cli') + $pfArgs)
    if ($LASTEXITCODE -ne 0) { throw "pf $($pfArgs -join ' ') failed with exit code $LASTEXITCODE" }
}

function Task-Ingest { Task-Pf (@('ingest') + $Rest) }
function Task-Eval   { Task-Pf (@('eval')   + $Rest) }
function Task-Cv     { Task-Pf (@('cv')     + $Rest) }
function Task-Audit  { Task-Pf (@('cameras', 'audit') + $Rest) }

function Task-Serve {
    $py = Get-Venv
    Write-Step 'API on http://127.0.0.1:8000  (docs at /docs)'
    & $py -m uvicorn parkfit.api.app:app --host 127.0.0.1 --port 8000 --reload
}

function Task-Web {
    Push-Location (Join-Path $Root 'web')
    try {
        if (-not (Test-Path 'node_modules')) {
            Write-Step 'npm install'
            & npm.cmd install
            if ($LASTEXITCODE -ne 0) { throw 'npm install failed' }
        }
        Write-Step 'Vite dev server on http://127.0.0.1:5173'
        & npm.cmd run dev
    } finally { Pop-Location }
}

function Task-Clean {
    foreach ($p in @($BuildDir, (Join-Path $Root '.pytest_cache'), (Join-Path $Root '.ruff_cache'), (Join-Path $Root '.mypy_cache'))) {
        if (Test-Path $p) { Remove-Item -Recurse -Force $p; Write-Ok "removed $p" }
    }
}

function Task-Help {
    Write-Host @'
CamToParkingSlot task runner

  setup     create the virtualenv, install dependencies, configure CMake
  build     compile the C++ core, vision worker and Python bindings
  rebuild   wipe the build directory and build from scratch
  test      run the C++ and Python suites
  ctest     C++ tests only
  pytest    Python tests only        (extra args pass through)
  lint      ruff + mypy
  format    ruff format and autofix
  ingest    pull open data into the local database
  serve     run the FastAPI backend
  web       run the Vite dev server for the progressive web app
  eval      print the accuracy metric table against its targets
  cv        run the camera vision worker
  audit     crawl and catalogue candidate camera sources
  clean     remove build and cache directories
'@ -ForegroundColor Gray
}

switch ($Task) {
    'setup'   { Task-Setup }
    'build'   { Task-Build }
    'rebuild' { Task-Rebuild }
    'test'    { Task-Test }
    'ctest'   { Task-CTest }
    'pytest'  { Task-Pytest }
    'lint'    { Task-Lint }
    'format'  { Task-Format }
    'ingest'  { Task-Ingest }
    'serve'   { Task-Serve }
    'web'     { Task-Web }
    'eval'    { Task-Eval }
    'cv'      { Task-Cv }
    'audit'   { Task-Audit }
    'clean'   { Task-Clean }
    default   { Task-Help }
}
