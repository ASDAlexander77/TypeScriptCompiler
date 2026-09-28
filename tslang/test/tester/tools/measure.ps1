<#
.SYNOPSIS
    Peak memory of one program under each memory model, ahead-of-time compiled.

.DESCRIPTION
    Builds a native exe per model (--emit=obj + lld, the same link line the test runner uses),
    runs it, and samples the process's peak working set until it exits. Prints one line per
    model: model, peak MB, exit code. Exits 1 if any run exited non-zero.

    Why AOT and not the JIT: under the JIT, 13-16 MB of every reading is tslang.exe itself, and
    each model's allocations are elided differently, so the same program read 41 MB one hour and
    12.6 MB the next. A native exe's floor is about 3 MB.

    Two things this must keep doing:
    - read the peak from the process handle after exit (GetProcessMemoryInfo), not by sampling
      while it runs: Process.PeakWorkingSet64 is unavailable once the process has exited, and a
      program that finishes in a few milliseconds is gone before the first sample - it read
      0.0 MB under every model;
    - the work directory is a fresh temp directory, never beside the source (the source is
      usually a repo file).

    The tool paths and the link line are read from the `compile.bat` the test runner generates
    in the build tree, so this links exactly the way the suite does.

.EXAMPLE
    ./measure.ps1 -Source ../own/own_fresh_string.ts
    ./measure.ps1 -Source raytrace.ts -Models gc,rc
#>
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [string[]]$Models = @('gc', 'rc', 'none', 'own'),
    [string]$BuildDir = (Join-Path $PSScriptRoot '../../../../__build/tslang/windows-msbuild-2026-release')
)

$ErrorActionPreference = 'Stop'

if (-not ('TslangMeasure.Psapi' -as [type])) {
    Add-Type -Namespace TslangMeasure -Name Psapi -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)]
public struct PROCESS_MEMORY_COUNTERS {
    public uint cb; public uint PageFaultCount;
    public UIntPtr PeakWorkingSetSize; public UIntPtr WorkingSetSize;
    public UIntPtr QuotaPeakPagedPoolUsage; public UIntPtr QuotaPagedPoolUsage;
    public UIntPtr QuotaPeakNonPagedPoolUsage; public UIntPtr QuotaNonPagedPoolUsage;
    public UIntPtr PagefileUsage; public UIntPtr PeakPagefileUsage;
}
[DllImport("psapi.dll", SetLastError = true)]
static extern bool GetProcessMemoryInfo(IntPtr process, out PROCESS_MEMORY_COUNTERS counters, uint size);
// Valid after the process has exited, for as long as the handle is open.
public static long PeakWorkingSet(IntPtr process) {
    PROCESS_MEMORY_COUNTERS counters;
    if (!GetProcessMemoryInfo(process, out counters, (uint)Marshal.SizeOf(typeof(PROCESS_MEMORY_COUNTERS))))
        throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
    return (long)counters.PeakWorkingSetSize.ToUInt64();
}
'@
}

$Source = (Resolve-Path $Source).Path
$bat = Join-Path $BuildDir 'test/tester/compile.bat'
if (-not (Test-Path $bat)) {
    throw "no $bat - run any test-runner compile test once (ctest -C Release -R test-compile-00) to generate it"
}

# `set NAME=value` lines, and the lld line with its %NAME% references expanded
$vars = @{}
$linkTemplate = $null
foreach ($line in Get-Content $bat) {
    if ($line -match '^set ([A-Z_]+)=(.*)$') {
        $vars[$Matches[1]] = $Matches[2].Trim('"')
    }
    elseif ($line -match 'lld\.exe -flavor link') {
        $linkTemplate = $line
    }
}
if (-not $linkTemplate) { throw "no lld line in $bat" }

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("tslang-measure-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null

$failed = $false
try {
    Push-Location $work
    foreach ($model in $Models) {
        $name = "m_$model"
        & "$($vars['TSLANGEXEPATH'])/tslang.exe" --emit=obj --entry-point --opt --opt_level=3 --no-default-lib "-mm=$model" $Source "-o=$name.obj"
        if ($LASTEXITCODE -ne 0) { Write-Output ("{0,-5} compile failed ({1})" -f $model, $LASTEXITCODE); $failed = $true; continue }

        $link = $linkTemplate -replace '%FILENAME%', $name -replace '%LINKER_OPTS%', ''
        foreach ($key in $vars.Keys) { $link = $link.Replace("%$key%", '"' + $vars[$key] + '"') }
        cmd /c $link | Out-Null
        if (-not (Test-Path "$name.exe")) { Write-Output ("{0,-5} link failed" -f $model); $failed = $true; continue }

        $startInfo = New-Object System.Diagnostics.ProcessStartInfo (Join-Path $work "$name.exe")
        $startInfo.UseShellExecute = $false
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = [System.Diagnostics.Process]::Start($startInfo)
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout.Result | Set-Content "$name.txt"
        $stderr.Result | Set-Content "$name.err"
        $peak = [TslangMeasure.Psapi]::PeakWorkingSet($process.Handle)
        $code = $process.ExitCode
        if ($code -ne 0) { $failed = $true }
        Write-Output ("{0,-5} {1,8:N1} MB  exit {2}" -f $model, ($peak / 1MB), $code)
    }
}
finally {
    Pop-Location
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

if ($failed) { exit 1 }
