# Install biotimbre and a shortcut for its browser interface, on Windows.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1 | Out-String | iex"
#
# Installs uv if it is missing (uv brings its own Python, so none is needed
# beforehand), installs the Microsoft Visual C++ Redistributable if this
# machine's is too old for the models, installs biotimbre as an isolated tool,
# and adds Desktop and Start menu shortcuts that open the interface from a work
# folder. Run it again to upgrade.
#
# Environment overrides:
#   BIOTIMBRE_PACKAGE   what to install                 (default: biotimbre[gui])
#   BIOTIMBRE_WORKDIR   folder the interface starts in  (default: %USERPROFILE%\biotimbre)
#   BIOTIMBRE_YES=1     install without asking, e.g. from a script or CI
#
# It says what it is about to do and asks before doing any of it.
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI.

$ErrorActionPreference = "Stop"

$Package = if ($env:BIOTIMBRE_PACKAGE) { $env:BIOTIMBRE_PACKAGE } else { "biotimbre[gui]" }
$WorkDir = if ($env:BIOTIMBRE_WORKDIR) { $env:BIOTIMBRE_WORKDIR } else { Join-Path $HOME "biotimbre" }
# The version CI tests on. Pinning it keeps a newer system Python, which
# onnxruntime may not publish wheels for yet, from being picked.
$Python = "3.12"

function Say($Message) { Write-Host "==> $Message" }

$Found = Get-Command uv -ErrorAction SilentlyContinue
$Uv = if ($Found) { $Found.Source } else { Join-Path $HOME ".local\bin\uv.exe" }
$HaveUv = Test-Path $Uv

# onnxruntime, which runs the models, is built with Visual Studio 2022 17.10 or
# later, and what that builds needs the 14.40 Visual C++ runtime or newer. With
# an older msvcp140.dll, importing onnxruntime fails or crashes the process
# outright. Windows does not update this runtime by itself, so a Windows 10
# machine can easily have one from years ago.
$VcMinimum = [Version]"14.40"
$VcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
# A 32-bit PowerShell sees SysWOW64 as System32; Sysnative is the real one.
$SystemDir = if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) { "Sysnative" } else { "System32" }
$VcDll = Join-Path $env:SystemRoot "$SystemDir\msvcp140.dll"
$VcVersion = $null
if (Test-Path $VcDll) {
    $Info = (Get-Item $VcDll).VersionInfo
    $VcVersion = New-Object Version $Info.FileMajorPart, $Info.FileMinorPart, $Info.FileBuildPart
}
$NeedVc = (-not $VcVersion) -or ($VcVersion -lt $VcMinimum)

# --- confirmation ------------------------------------------------------------
$UvText = if ($HaveUv) { "use uv, already installed at $Uv" } else { "install uv, the tool that installs biotimbre, into $(Join-Path $HOME '.local\bin')" }
$VcText = ""
if ($NeedVc) {
    $VcFound = if ($VcVersion) { "this computer has $VcVersion" } else { "this computer has none" }
    $VcText = @"
  - install the Microsoft Visual C++ Redistributable (x64), from Microsoft:
    the models need version $VcMinimum or newer, and $VcFound
    (Windows asks for an administrator's permission)

"@
}
Write-Host @"
This installs biotimbre, a tool for building and searching libraries of
animal sounds, with its interface in the browser. It will:

  - $UvText
$VcText  - install $Package with uv, isolated, along with a Python $Python
    of its own; your other Pythons are not touched (if biotimbre is already
    installed, it is upgraded)
  - add uv's tools folder to your user PATH
  - create the folder $WorkDir, where the interface starts
  - add biotimbre shortcuts, with its icon, to your Desktop and Start menu

The first start of the interface downloads the models (about 70 MB), once.

"@
if ($env:BIOTIMBRE_YES -ne "1") {
    if ([Console]::IsInputRedirected) {
        throw "there is no console to ask in; set BIOTIMBRE_YES=1 to install without asking."
    }
    $Answer = Read-Host "Continue? [y/N]"
    if ($Answer -notmatch '^\s*(y|yes)\s*$') {
        Write-Host "Nothing was installed."
        # Ends this script only, also when it runs through iex in an open
        # PowerShell, where exit would close the window.
        return
    }
}

# --- uv ----------------------------------------------------------------------
if (-not $HaveUv) {
    Say "Installing uv"
    $env:UV_INSTALL_DIR = Join-Path $HOME ".local\bin"
    # A separate process, so an exit in uv's installer cannot end this one.
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Uv)) { throw "uv could not be installed" }
}

# --- Visual C++ runtime -----------------------------------------------------
# A failure here is reported but does not stop the install: biotimbre works
# without the models until the runtime is installed, and says so when they fail.
$VcProblem = $null
$VcRestart = $false
if ($NeedVc) {
    Say "Installing the Microsoft Visual C++ Redistributable"
    $VcInstaller = Join-Path $env:TEMP "vc_redist.x64.exe"
    try {
        # 5.1 may not offer TLS 1.2 unless asked, and its progress bar makes
        # the download many times slower.
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $SavedProgress = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $VcUrl -OutFile $VcInstaller
        } finally {
            $ProgressPreference = $SavedProgress
        }
        # Its own window shows progress. RunAs is what asks for permission.
        $Proc = Start-Process -FilePath $VcInstaller -ArgumentList "/install", "/passive", "/norestart" -Verb RunAs -PassThru
        # Taking the handle before waiting is what keeps ExitCode readable
        # for an elevated process; with -Wait it can come back empty.
        $null = $Proc.Handle
        $Proc.WaitForExit()
        # 1638: a newer one is already installed. 3010: installed, but files
        # in use are only replaced at the next restart.
        if ($Proc.ExitCode -eq 3010) {
            $VcRestart = $true
        } elseif ($Proc.ExitCode -notin 0, 1638) {
            $VcProblem = "its installer exited with code $($Proc.ExitCode)"
        }
    } catch {
        $VcProblem = "$($_.Exception.Message)"
    } finally {
        Remove-Item -Force $VcInstaller -ErrorAction SilentlyContinue
    }
    if ($VcProblem) {
        Write-Warning "The Visual C++ Redistributable was not installed ($VcProblem). biotimbre is installed anyway, but its models will not load until it is: install it from $VcUrl"
    }
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

# --- icon --------------------------------------------------------------------
# Copied out of the tool's environment, which uv may rebuild on an upgrade, to
# a place that stays put for as long as the shortcuts point at it. A package
# from before the logo has none, and the shortcuts then keep the default icon.
$ToolDir = (& $Uv tool dir | Out-String).Trim()
$PackagedIcon = Join-Path $ToolDir "biotimbre\Lib\site-packages\biotimbre\assets\biotimbre.ico"
$Icon = $null
if (Test-Path $PackagedIcon) {
    $IconDir = Join-Path $env:LOCALAPPDATA "biotimbre"
    New-Item -ItemType Directory -Force -Path $IconDir | Out-Null
    $Icon = Join-Path $IconDir "biotimbre.ico"
    Copy-Item -Force $PackagedIcon $Icon
}

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
    if ($Icon) { $Link.IconLocation = "$Icon,0" }
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
if ($VcRestart) {
    Write-Host ""
    Write-Host "  Restart Windows before the first start: the Visual C++ runtime that"
    Write-Host "  was just installed only takes effect after a restart."
}
if ($VcProblem) {
    Write-Host ""
    Write-Host "  The models need the Microsoft Visual C++ Redistributable, which was"
    Write-Host "  not installed. Install it from $VcUrl"
}
