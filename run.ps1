<#
.SYNOPSIS
Automatically set up and run the OpenAI-only Persian voice application.
.EXAMPLE
.\run.ps1
.EXAMPLE
.\run.ps1 -SkipInstall -NoBrowser -Port 8501
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8500,
    [string]$BindAddress = '0.0.0.0',
    [switch]$EnableLocal,
    [switch]$SkipInstall,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$runnerExitCode = 1
Push-Location $PSScriptRoot
try {
    $runnerArguments = @((Join-Path $PSScriptRoot 'run.py'), '--host', $BindAddress, '--port', "$Port")
    if ($EnableLocal) { $runnerArguments += '--enable-local' }
    if ($SkipInstall) { $runnerArguments += '--skip-install' }
    if ($NoBrowser) { $runnerArguments += '--no-browser' }
    $windowsPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $unixPython = Join-Path $PSScriptRoot '.venv/bin/python'
    if (Test-Path $windowsPython) { & $windowsPython @runnerArguments }
    elseif (Test-Path $unixPython) { & $unixPython @runnerArguments }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3 @runnerArguments }
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) { & python3 @runnerArguments }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python @runnerArguments }
    else { throw 'Install Python 3.12 first, then run this launcher again.' }
    $runnerExitCode = $LASTEXITCODE
}
catch { Write-Host $_.Exception.Message -ForegroundColor Red; $runnerExitCode = 1 }
finally { Pop-Location }
exit $runnerExitCode
