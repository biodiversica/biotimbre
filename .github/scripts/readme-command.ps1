# Run one of the README's Windows commands as typed, against this checkout's
# install.ps1 served the way a GitHub release serves it.
#
# Release assets are application/octet-stream, and Windows PowerShell 5.1's
# irm hands such a script to iex line by line. So a local server sends this
# checkout's script the same way, and the command's URL is pointed at it: what
# runs is the README's command, quoting and all, on the script being tested.
#
#   powershell -NoProfile -ExecutionPolicy ByPass -File .github\scripts\readme-command.ps1 [-Uninstall]

param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$Release = "https://github.com/biodiversica/biotimbre/releases/latest/download/install.ps1"
$Local = "http://localhost:8766/install.ps1"

$Server = Start-Job -ArgumentList (Resolve-Path scripts\install.ps1).Path {
    param($Path)
    $Bytes = [IO.File]::ReadAllBytes($Path)
    $Listener = New-Object System.Net.HttpListener
    $Listener.Prefixes.Add("http://localhost:8766/")
    $Listener.Start()
    while ($true) {
        $Context = $Listener.GetContext()
        $Context.Response.ContentType = "application/octet-stream"
        $Context.Response.OutputStream.Write($Bytes, 0, $Bytes.Length)
        $Context.Response.Close()
    }
}
try {
    $Ready = $false
    foreach ($i in 1..30) {
        try { $null = Invoke-RestMethod $Local; $Ready = $true; break } catch { Start-Sleep 1 }
    }
    if (-not $Ready) { Receive-Job $Server; throw "the local server did not start" }

    # What irm returns, here and from the real release, for the log.
    foreach ($Url in $Local, $Release) {
        try {
            $Body = Invoke-RestMethod $Url
            Write-Host ("irm {0} -> {1}, {2} item(s)" -f $Url, $Body.GetType().FullName, @($Body).Count)
        } catch { Write-Host "irm $Url failed: $_" }
    }

    # The README has two such lines; the uninstall one sets BIOTIMBRE_UNINSTALL.
    $Which = if ($Uninstall) { "uninstall" } else { "install" }
    $Line = Select-String -Path README.md -SimpleMatch "irm $Release" |
        ForEach-Object { $_.Line } |
        Where-Object { ($_ -match "BIOTIMBRE_UNINSTALL") -eq [bool]$Uninstall } |
        Select-Object -First 1
    if (-not $Line) { throw "README.md has no $Which command fetching $Release" }
    $Command = $Line.Replace($Release, $Local)
    Write-Host "> $Command"
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) { throw "the README's $Which command failed with exit code $LASTEXITCODE" }
} finally {
    Stop-Job $Server
    Remove-Job $Server -Force
}
