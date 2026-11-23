# Install biotimbre and a shortcut for its browser interface, on Windows.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs biotimbre as an isolated tool, and adds Desktop and
# Start menu shortcuts that open the interface from a work folder. Run it again
# to upgrade.
#
# To uninstall, run it with BIOTIMBRE_UNINSTALL set (single quotes, so the
# PowerShell it is typed into leaves $env alone):
#
#   powershell -ExecutionPolicy ByPass -c '$env:BIOTIMBRE_UNINSTALL=1; irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex'
#
# Environment overrides:
#   BIOTIMBRE_PACKAGE    what to install                 (default: biotimbre[gui])
#   BIOTIMBRE_WORKDIR    folder the interface starts in  (default: %USERPROFILE%\biotimbre)
#   BIOTIMBRE_UNINSTALL  1 removes biotimbre and its shortcuts; all also removes
#                        the downloaded models and the interface's remembered
#                        settings. Libraries are never removed, nor is uv.
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

$ErrorActionPreference = "Stop"

$Package = if ($env:BIOTIMBRE_PACKAGE) { $env:BIOTIMBRE_PACKAGE } else { "biotimbre[gui]" }
$WorkDir = if ($env:BIOTIMBRE_WORKDIR) { $env:BIOTIMBRE_WORKDIR } else { Join-Path $HOME "biotimbre" }
$Uninstall = $env:BIOTIMBRE_UNINSTALL
# The version CI tests on. Pinning it keeps a newer system Python, which
# onnxruntime may not publish wheels for yet, from being picked.
$Python = "3.12"
# Where the shortcuts go, and so where uninstalling looks for them.
$Places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
# The models' Hugging Face repositories, as the hub cache names their folders.
$ModelFolders = @("models--biodiversica--Perch-onnx-backbone", "models--biodiversica--BirdNET-onnx-backbone")

function Say($Message) { Write-Host "==> $Message" }

function Find-Uv {
    $Found = Get-Command uv -ErrorAction SilentlyContinue
    if ($Found) { $Found.Source } else { Join-Path $HOME ".local\bin\uv.exe" }
}

function Remove-IfPresent($Path, $What) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
        Say "Removed $What ($Path)"
    }
}

function Uninstall-Biotimbre {
    if ($Uninstall -notin @("1", "all")) {
        throw "BIOTIMBRE_UNINSTALL must be 1 or all, not '$Uninstall'"
    }
    # Windows will not delete a program that is running.
    if (Get-Process -Name biotimbre -ErrorAction SilentlyContinue) {
        throw "biotimbre is running. Close its window (or press Ctrl+C in it), then run this again."
    }

    $Uv = Find-Uv
    if ((Test-Path $Uv) -and ((& $Uv tool list | Out-String) -match "(?m)^biotimbre ")) {
        Say "Uninstalling biotimbre"
        & $Uv tool uninstall biotimbre
        if ($LASTEXITCODE -ne 0) { throw "uv tool uninstall failed with exit code $LASTEXITCODE" }
    }
    foreach ($Folder in $Places) {
        Remove-IfPresent (Join-Path $Folder "biotimbre.lnk") "shortcut"
    }

    if ($Uninstall -eq "all") {
        $Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
        Remove-IfPresent (Join-Path $Local "biotimbre") "remembered settings"
        # Only biotimbre's own models: the cache is shared with other programs.
        $Hub = Join-Path $HOME ".cache\huggingface\hub"
        if ($env:HF_HOME) { $Hub = Join-Path $env:HF_HOME "hub" }
        if ($env:HF_HUB_CACHE) { $Hub = $env:HF_HUB_CACHE }
        foreach ($Name in $ModelFolders) {
            Remove-IfPresent (Join-Path $Hub $Name) "model"
            Remove-IfPresent (Join-Path (Join-Path $Hub ".locks") $Name) "model lock"
        }
    }

    Say "Done. biotimbre is uninstalled."
    Write-Host ""
    Write-Host "  Libraries were left where they are, including any in $WorkDir."
    Write-Host "  Delete them yourself if you no longer want them."
}

function Install-Biotimbre {
    # --- uv ------------------------------------------------------------------
    $Uv = Find-Uv
    if (-not (Test-Path $Uv)) {
        Say "Installing uv"
        $env:UV_INSTALL_DIR = Join-Path $HOME ".local\bin"
        # A separate process, so an exit in uv's installer cannot end this one.
        powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Uv)) { throw "uv could not be installed" }
    }

    # --- biotimbre -----------------------------------------------------------
    Say "Installing $Package"
    # Native commands do not stop on failure under $ErrorActionPreference, and
    # their stderr must not be redirected: 5.1 would turn uv's progress lines
    # into errors.
    & $Uv tool install --python $Python --upgrade $Package
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed with exit code $LASTEXITCODE" }
    & $Uv tool update-shell
    $Bin = (& $Uv tool dir --bin | Out-String).Trim()
    $Exe = Join-Path $Bin "biotimbre.exe"
    if (-not (Test-Path $Exe)) { throw "biotimbre was not installed where uv said it would be ($Exe)" }

    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

    # --- shortcuts -----------------------------------------------------------
    # They open a console window: it shows the address and is where Ctrl+C
    # stops the interface.
    $Shell = New-Object -ComObject WScript.Shell
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
}

# Functions rather than an early return: under iex, a return at the top level
# does not reliably end the script.
if ($Uninstall) { Uninstall-Biotimbre } else { Install-Biotimbre }
