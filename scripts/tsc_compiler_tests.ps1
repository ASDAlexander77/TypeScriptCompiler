<#
.SYNOPSIS
    Compiles TypeScript's own compiler tests (tests/cases/compiler) with tslang and reports the
    files that do not compile.

.DESCRIPTION
    Each test file is split into units the way TypeScript's test harness does it:
      * a `// @filename: name` line starts a new unit named `name`;
      * every `// @option: value` directive line is dropped from the code;
      * text before the first `// @filename` goes into the first unit;
      * a test without `// @filename` is one unit named after the test file.
    The units are written into a temp folder per test (subfolders kept, so imports like
    "./a" or "node_modules/x/index.d.ts" resolve) and every .ts/.tsx/.mts/.cts unit is compiled
    with `tslang --emit=obj <unit> -o <unit>.o` (tslang takes one input and follows imports
    itself). A test passes when all of its units compile.

    Results: <OutDir>\results.csv (one row per test), <OutDir>\failed.txt (failing test names),
    <OutDir>\logs\<test>.log (compiler output of failing tests).

.EXAMPLE
    pwsh scripts\tsc_compiler_tests.ps1
.EXAMPLE
    pwsh scripts\tsc_compiler_tests.ps1 -Filter 'alias*' -Jobs 4
.EXAMPLE
    pwsh scripts\tsc_compiler_tests.ps1 -OnlyTscClean -OutDir I:\temp\sweep2
#>
[CmdletBinding()]
param(
    # tests/cases/compiler of a clone of https://github.com/microsoft/TypeScript (the baselines for
    # -OnlyTscClean are read from tests/baselines/reference/compiler beside it)
    [string]$TestsDir = 'E:\Dev3\TypeScript\tsc\testdata\tests\cases\compiler',
    [string]$Tslang = (Join-Path $PSScriptRoot '..\__build\tslang\windows-msbuild-2026-release\bin\tslang.exe'),
    [string]$OutDir = (Join-Path ([IO.Path]::GetTempPath()) 'tslang-tsc-compiler-tests'),
    # Wildcard on the test file name, e.g. 'class*'
    [string]$Filter = '*',
    [int]$Jobs = 8,
    [int]$TimeoutSec = 60,
    # Extra arguments for every tslang call, e.g. @('--mm=own')
    [string[]]$ExtraArgs = @(),
    # Only tests TypeScript itself compiles without errors (no baselines/reference/compiler/<name>.errors.txt)
    [switch]$OnlyTscClean,
    # Also compile .d.ts units on their own (they are still written and can be imported either way)
    [switch]$IncludeDts,
    # Keep the temp folder of each test (by default only failing tests keep theirs)
    [switch]$KeepAll,
    # Stop after this many tests (0 = all)
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'

$Tslang = (Resolve-Path $Tslang).Path
if (-not (Test-Path $TestsDir)) { throw "Tests folder not found: $TestsDir" }

# No Windows error / crash dialogs for crashing compiler runs (inherited by child processes).
Add-Type -Namespace Native -Name Kernel32 -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetErrorMode(uint mode);
'@
[void][Native.Kernel32]::SetErrorMode(0x0001 -bor 0x0002 -bor 0x8000) # FAILCRITICALERRORS | NOGPFAULTERRORBOX | NOOPENFILEERRORBOX

$workDir = Join-Path $OutDir 'work'
$logDir = Join-Path $OutDir 'logs'
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force $workDir, $logDir | Out-Null

$tests = Get-ChildItem -Path $TestsDir -File | Where-Object { $_.Extension -in '.ts', '.tsx' -and $_.Name -like $Filter } | Sort-Object Name
if ($OnlyTscClean) {
    $baselineDir = Join-Path $TestsDir '..\..\..\baselines\reference\compiler'
    $tests = $tests | Where-Object { -not (Test-Path (Join-Path $baselineDir ($_.BaseName + '.errors.txt'))) }
}
if ($Limit -gt 0) { $tests = $tests | Select-Object -First $Limit }
$total = @($tests).Count
Write-Host "tslang: $Tslang"
Write-Host "Tests : $total from $TestsDir ($Jobs jobs, ${TimeoutSec}s timeout)"
Write-Host "Output: $OutDir"

$started = Get-Date
$results = $tests | ForEach-Object -ThrottleLimit $Jobs -Parallel {
    $test = $_
    $tslang = $using:Tslang
    $workDir = $using:workDir
    $logDir = $using:logDir
    $timeoutMs = $using:TimeoutSec * 1000
    $extraArgs = $using:ExtraArgs
    $includeDts = $using:IncludeDts
    $keepAll = $using:KeepAll

    # --- split into units (same rules as TypeScript's harness) ---
    $directive = '^\/{2}\s*@(\w+)\s*:\s*([^\r\n]*)'
    $units = [System.Collections.Generic.List[object]]::new()
    $curName = $null
    $curText = $null
    foreach ($line in [IO.File]::ReadAllLines($test.FullName)) {
        if ($line -match $directive) {
            if ($Matches[1].ToLowerInvariant() -eq 'filename') {
                if ($curName) { $units.Add([pscustomobject]@{ Name = $curName; Text = $curText }); $curText = $null }
                $curName = $Matches[2].Trim()
            }
            continue
        }
        if ($null -eq $curText) { $curText = $line } else { $curText += "`n" + $line }
    }
    if (-not $curName) { $curName = $test.Name }
    $units.Add([pscustomobject]@{ Name = $curName; Text = $curText })

    # --- write units ---
    $dir = Join-Path $workDir $test.BaseName
    New-Item -ItemType Directory -Force $dir | Out-Null
    $toCompile = @()
    foreach ($u in $units) {
        # "/a/b.ts", "c:/x.ts", "../x.ts" -> keep it inside the test folder
        $rel = ($u.Name -replace '\\', '/') -replace '^[A-Za-z]:', '' -replace '^/+', '' -replace '(^|/)\.\.(?=/)', '$1__up'
        $path = Join-Path $dir $rel
        New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
        $text = [string]$u.Text
        # The harness serves tests/lib as "/.lib": copy it in and point the references at it
        if ($text.Contains('/.lib/')) {
            $libDir = Join-Path $dir '.lib'
            if (-not (Test-Path $libDir)) { Copy-Item -Recurse (Join-Path $using:TestsDir '..\..\lib') $libDir }
            $depth = $rel.Split('/').Count - 1
            $prefix = if ($depth -eq 0) { './' } else { '../' * $depth }
            $text = $text.Replace('/.lib/', "$prefix.lib/")
        }
        [IO.File]::WriteAllText($path, $text)
        $isCode = $rel -match '\.(ts|tsx|mts|cts)$'
        $isDts = $rel -match '\.d\.(ts|mts|cts)$' -or $rel -match '\.d\.[^/]*\.ts$'
        if ($isCode -and ($includeDts -or -not $isDts)) { $toCompile += $path }
    }

    # --- compile each unit ---
    $status = 'pass'
    $failedUnits = @()
    $firstError = ''
    $log = [Text.StringBuilder]::new()
    $sw = [Diagnostics.Stopwatch]::StartNew()
    if ($toCompile.Count -eq 0) { $status = 'skipped' }
    foreach ($path in $toCompile) {
        $psi = [Diagnostics.ProcessStartInfo]::new($tslang)
        $psi.WorkingDirectory = $dir
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true
        $psi.ArgumentList.Add('--emit=obj')
        foreach ($a in $extraArgs) { $psi.ArgumentList.Add($a) }
        $psi.ArgumentList.Add($path)
        $psi.ArgumentList.Add('-o')
        $psi.ArgumentList.Add("$path.o")
        $p = [Diagnostics.Process]::Start($psi)
        $outTask = $p.StandardOutput.ReadToEndAsync()
        $errTask = $p.StandardError.ReadToEndAsync()
        $unitStatus = 'pass'
        if (-not $p.WaitForExit($timeoutMs)) {
            try { $p.Kill($true) } catch {}
            $p.WaitForExit()
            $unitStatus = 'timeout'
        }
        else {
            $p.WaitForExit() # flush async readers
            $code = $p.ExitCode
            if ($code -ne 0) {
                # NTSTATUS exception codes (0xC0000005 access violation, 0xC00000FD stack overflow, 0xE06D7363 C++
                # exception, 0x80000003 breakpoint) and abort() (3) are crashes; -1 is tslang's own error exit
                $unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$code), 0)
                $isException = ($unsigned -ge [uint32]'0xC0000000' -and $unsigned -ne [uint32]'0xFFFFFFFF') -or $unsigned -eq [uint32]'0x80000003'
                $unitStatus = if ($isException -or $code -eq 3) { 'crash' } else { 'fail' }
            }
        }
        $output = ($outTask.Result + $errTask.Result).Trim()
        if ($unitStatus -ne 'pass') {
            $failedUnits += (Split-Path $path -Leaf)
            $code = if ($unitStatus -eq 'timeout') { 'timeout' } else { '0x{0:X8}' -f $p.ExitCode }
            [void]$log.AppendLine("===== $path [$unitStatus, exit $code]")
            [void]$log.AppendLine($output)
            if (-not $firstError) {
                $firstError = ($output -split "`r?`n" | Where-Object { $_ -match 'error' } | Select-Object -First 1)
                if (-not $firstError) { $firstError = ($output -split "`r?`n" | Select-Object -First 1) }
                if (-not $firstError) { $firstError = "<$unitStatus, no output>" }
            }
            # worst status wins: crash > timeout > fail
            $rank = @{ pass = 0; fail = 1; timeout = 2; crash = 3 }
            if ($rank[$unitStatus] -gt $rank[$status]) { $status = $unitStatus }
        }
        $p.Dispose()
    }
    $sw.Stop()

    if ($status -in 'fail', 'crash', 'timeout') {
        [IO.File]::WriteAllText((Join-Path $logDir ($test.BaseName + '.log')), $log.ToString())
    }
    elseif (-not $keepAll) {
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    }

    [pscustomobject]@{
        Test        = $test.Name
        Status      = $status
        Units       = $units.Count
        Compiled    = $toCompile.Count
        FailedUnits = ($failedUnits -join ';')
        Seconds     = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        FirstError  = ([string]$firstError).Replace($dir + '\', '').Trim()
    }
}

$results = @($results | Sort-Object Test)
$results | Export-Csv -Encoding utf8 (Join-Path $OutDir 'results.csv')
$bad = @($results | Where-Object Status -in 'fail', 'crash', 'timeout')
$bad | ForEach-Object Test | Set-Content -Encoding utf8 (Join-Path $OutDir 'failed.txt')

# Failing tests grouped by their first error (names in quotes -> 'X', numbers -> N)
$groups = $bad | Group-Object {
    if ($_.Status -ne 'fail') { "[$($_.Status)] " + ($(if ($_.FirstError -like '<*') { '<no output>' } else { $_.FirstError -replace '^.*?error:\s*', '' })) }
    else { $_.FirstError -replace '^.*?error:\s*', '' }
} | ForEach-Object {
    $key = $_.Name -replace "(?<![A-Za-z])'[^']*'(?![A-Za-z])", "'X'" -replace '"[^"]*"', '"X"' `
        -replace '((?:name|file|named|symbol):\s*)\S+', '$1X' -replace '\d+', 'N'
    [pscustomobject]@{ Key = $key; Items = $_.Group }
} | Group-Object Key | Sort-Object { ($_.Group.Items).Count } -Descending
$md = [Text.StringBuilder]::new()
[void]$md.AppendLine("# tslang: failing tsc compiler tests by error ($($bad.Count) tests, $(@($groups).Count) groups)`n")
[void]$md.AppendLine('| Tests | Error |')
[void]$md.AppendLine('|---:|---|')
foreach ($g in $groups) { [void]$md.AppendLine("| $(@($g.Group.Items).Count) | ``$($g.Name -replace '\|', '\|')`` |") }
foreach ($g in $groups) {
    $items = @($g.Group.Items) | Sort-Object Test
    [void]$md.AppendLine("`n## $($items.Count) - ``$($g.Name)```n")
    foreach ($it in $items) {
        $msg = $it.FirstError -replace '\s+', ' '
        [void]$md.AppendLine("- **$($it.Test)** ($($it.FailedUnits)): $msg")
    }
}
[IO.File]::WriteAllText((Join-Path $OutDir 'by-error.md'), $md.ToString())

$elapsed = (Get-Date) - $started
Write-Host ''
Write-Host ("Done in {0:hh\:mm\:ss}" -f $elapsed)
$results | Group-Object Status | Sort-Object Name | ForEach-Object { Write-Host ("  {0,-8} {1}" -f $_.Name, $_.Count) }
$multi = @($results | Where-Object Units -gt 1)
Write-Host ("  multi-file tests: {0}, failing: {1}" -f $multi.Count, @($multi | Where-Object Status -in 'fail', 'crash', 'timeout').Count)
Write-Host ''
if ($bad.Count) {
    Write-Host "Files that can't be compiled ($($bad.Count)):"
    $bad | Sort-Object Status, Test | Format-Table Status, Test, FailedUnits, FirstError -AutoSize -Wrap | Out-String -Width 250 | Write-Host
}
Write-Host "Results : $(Join-Path $OutDir 'results.csv')"
Write-Host "Failed  : $(Join-Path $OutDir 'failed.txt')"
Write-Host "By error: $(Join-Path $OutDir 'by-error.md')"
Write-Host "Logs    : $logDir"
