# Samples the call stacks of the Release compiler while it compiles one file, and prints the
# functions the time is in. Windows, no administrator rights needed.
#
#   pwsh tslang/benchmarks/compile-time/profile.ps1 <source.ts> [-Flags --mm=rc] [-Interval 1]
#
# What it does:
#  1. relinks the Release tslang.exe with symbols into out/ (tslang_prof.exe + .pdb), from the link
#     command MSBuild recorded for it: the build directory is not touched, and nothing is rebuilt;
#  2. builds sampler.cpp (cl, from the newest Visual Studio);
#  3. runs the compiler under the sampler and writes out/profile.txt: exclusive and inclusive
#     percentages per function, and the commonest stack of the top leaves.
#
# The numbers are for locating code inside one stage, not for comparing stages: walking a deep
# stack takes longer, so the stages with deep stacks (MLIR generation, the passes) get fewer
# samples than their time. --mlir-timing is what says how long each stage takes. A per-function
# MLIR pass runs on a worker thread, which this does not sample: add -Flags --mlir-disable-threading.
param(
    [Parameter(Mandatory)][string]$Source,
    [string[]]$Flags = @(),
    [int]$Interval = 1,
    [string]$BuildDir = "$PSScriptRoot\..\..\..\__build\tslang\windows-msbuild-2026-release",
    [switch]$Relink
)

$ErrorActionPreference = "Stop"
$out = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force $out | Out-Null
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -prerelease -products * -property installationPath
$vcvars = "$vs\VC\Auxiliary\Build\vcvars64.bat"

$exe = Join-Path $out "tslang_prof.exe"
$built = Join-Path $BuildDir "bin\tslang.exe"
if ($Relink -or -not (Test-Path $exe) -or (Get-Item $exe).LastWriteTime -lt (Get-Item $built).LastWriteTime) {
    $tlog = Join-Path $BuildDir "tslang\tslang.dir\Release\tslang.tlog\link.command.1.tlog"
    $cmd = ((Get-Content $tlog) | Where-Object { $_ -notmatch '^\^' }) -join ' '
    $cmd = $cmd -replace '/OUT:"[^"]*"', "/OUT:`"$exe`""
    $cmd = $cmd -replace '/PDB:"[^"]*"', "/PDB:`"$out\tslang_prof.pdb`""
    $cmd = $cmd -replace '/IMPLIB:"[^"]*"', "/IMPLIB:`"$out\tslang_prof.lib`""
    $cmd = $cmd -replace '/ERRORREPORT:\S+', ''
    $cmd += ' /DEBUG:FULL /OPT:REF /OPT:NOICF'
    Set-Content "$out\link.rsp" $cmd -Encoding Unicode
    Push-Location (Join-Path $BuildDir "tslang")
    try { cmd /c "`"$vcvars`" >nul && link @`"$out\link.rsp`"" 2>&1 | Select-Object -Last 3 } finally { Pop-Location }
}

$sampler = Join-Path $out "sampler.exe"
if (-not (Test-Path $sampler)) {
    Push-Location $out
    try { cmd /c "`"$vcvars`" >nul && cl /nologo /std:c++17 /O2 /EHsc `"$PSScriptRoot\sampler.cpp`" /Fe:sampler.exe dbghelp.lib" 2>&1 | Select-Object -Last 3 } finally { Pop-Location }
}

& $sampler $Interval $exe --emit=obj --no-default-lib @Flags $Source -o "$out\profiled.obj" > "$out\profile.txt" 2>&1
"written $out\profile.txt"
Get-Content "$out\profile.txt" -TotalCount 60
