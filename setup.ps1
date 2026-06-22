# One-time setup for KinectFX on Windows (PowerShell).
#   Right-click > Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Creating virtualenv (.venv)"
$py = "py"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { $py = "python" }
& $py -3.11 -m venv .venv 2>$null
if (-not (Test-Path ".venv")) { & $py -m venv .venv }

Write-Host "==> Installing Python dependencies"
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host @"

==> Python deps installed.

Two things still need a manual one-time setup on Windows:

  1. KINECT v1 USB DRIVER + libfreenect
     - Install the libusbK driver for the Kinect with Zadig (https://zadig.akeo.ie):
       run Zadig, Options > 'List All Devices', and for EACH of
       'Xbox NUI Camera', 'Xbox NUI Motor', 'Xbox NUI Audio' replace the driver
       with libusbK.
     - Put freenect_sync.dll and its dependencies (freenect.dll, libusb-1.0.dll,
       pthreadVC2.dll) in a 'libs' folder here, OR set KINECTFX_LIB_DIR to where
       they live. Prebuilt DLLs come from a libfreenect Windows build (CMake +
       libusb). Without these the app runs on a synthetic source.

  2. VIRTUAL CAMERA
     - Install OBS Studio, run it once and click 'Start Virtual Camera'
       (registers the OBS Virtual Camera), then quit OBS.

Then run:   .\run.bat      (or: .venv\Scripts\python.exe main.py)
In Discord: Settings > Voice & Video > Camera > 'OBS Virtual Camera'.
"@
