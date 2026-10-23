# Install biotimbre and a shortcut for its browser interface, on Windows.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | iex"
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs biotimbre as an isolated tool, and adds Desktop and
# Start menu shortcuts that open the interface from a work folder. Run it again
# to upgrade.
#
# Environment overrides:
#   BIOTIMBRE_PACKAGE   what to install                 (default: biotimbre[gui])
#   BIOTIMBRE_WORKDIR   folder the interface starts in  (default: %USERPROFILE%\biotimbre)
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

$ErrorActionPreference = "Stop"

$Package = if ($env:BIOTIMBRE_PACKAGE) { $env:BIOTIMBRE_PACKAGE } else { "biotimbre[gui]" }
$WorkDir = if ($env:BIOTIMBRE_WORKDIR) { $env:BIOTIMBRE_WORKDIR } else { Join-Path $HOME "biotimbre" }
# The version CI tests on. Pinning it keeps a newer system Python, which
# onnxruntime may not publish wheels for yet, from being picked.
$Python = "3.12"

function Say($Message) { Write-Host "==> $Message" }

# --- uv ----------------------------------------------------------------------
$Found = Get-Command uv -ErrorAction SilentlyContinue
$Uv = if ($Found) { $Found.Source } else { Join-Path $HOME ".local\bin\uv.exe" }
if (-not (Test-Path $Uv)) {
    Say "Installing uv"
    $env:UV_INSTALL_DIR = Join-Path $HOME ".local\bin"
    # A separate process, so an exit in uv's installer cannot end this one.
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Uv)) { throw "uv could not be installed" }
}

# --- biotimbre ---------------------------------------------------------------
Say "Installing $Package"
# Native commands do not stop on failure under $ErrorActionPreference, and their
# stderr must not be redirected: 5.1 would turn uv's progress lines into errors.
& $Uv tool install --python $Python --upgrade $Package
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed with exit code $LASTEXITCODE" }
& $Uv tool update-shell
$Bin = (& $Uv tool dir --bin | Out-String).Trim()
$Exe = Join-Path $Bin "biotimbre.exe"
if (-not (Test-Path $Exe)) { throw "biotimbre was not installed where uv said it would be ($Exe)" }

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# --- shortcuts ---------------------------------------------------------------
# They open a console window: it shows the address and is where Ctrl+C stops
# the interface.
$Shell = New-Object -ComObject WScript.Shell
$Places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
foreach ($Folder in $Places) {
    $Link = $Shell.CreateShortcut((Join-Path $Folder "biotimbre.lnk"))
    $Link.TargetPath = $Exe
    $Link.Arguments = "gui"
    $Link.WorkingDirectory = $WorkDir
    $Link.Description = "Build and search libraries of animal sounds in the browser"
    $Link.Save()
}

$Version = (& $Exe --version | Out-String).Trim()
Say "Done. $Version is installed."
Write-Host ""
Write-Host "  To start: double-click biotimbre on your Desktop or in the Start menu,"
Write-Host "  or run 'biotimbre gui' in a new terminal."
Write-Host ""
Write-Host "  The interface starts in $WorkDir, where it offers to create or open a"
Write-Host "  library. The first start downloads the models (about 70 MB), once."
Write-Host "  To upgrade later, run this installer again."
