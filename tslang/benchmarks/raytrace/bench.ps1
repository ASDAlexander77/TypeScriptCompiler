# Builds rtbench.ts under every memory model (with and without stack promotion) and the two Rust
# ports, checks that they all print the same checksum, then times each one: a warm-up run and
# $Runs timed runs, reporting min/median/max wall time.
#
#   pwsh tslang/benchmarks/raytrace/bench.ps1 [-Tslang <path to tslang.exe>] [-Runs 7]
param(
    [string]$Tslang = "$PSScriptRoot\..\..\..\__build\tslang\windows-msbuild-2026-release\bin\tslang.exe",
    [int]$Runs = 7
)

$ErrorActionPreference = "Stop"
$out = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force $out | Out-Null

$bins = @()
foreach ($mm in "gc", "rc", "none", "own") {
    foreach ($sp in $false, $true) {
        $name = if ($sp) { "tslang $mm (no SP)" } else { "tslang $mm" }
        $exe = Join-Path $out ("rtb_$mm" + $(if ($sp) { "_nsp" } else { "" }) + ".exe")
        $flags = @("--emit=exe", "--opt", "--opt_level=3", "--no-default-lib", "--mm=$mm")
        if ($sp) { $flags += "--no-stack-promotion" }
        & $Tslang @flags "$PSScriptRoot\rtbench.ts" -o $exe 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { "{0,-22} does not compile (exit {1})" -f $name, $LASTEXITCODE; continue }
        $bins += @{ n = $name; p = $exe }
    }
}

Push-Location "$PSScriptRoot\rust"
try { cargo build --release --quiet; if ($LASTEXITCODE -ne 0) { throw "cargo build failed" } } finally { Pop-Location }
$bins += @{ n = "rust idiomatic"; p = "$PSScriptRoot\rust\target\release\idiomatic.exe" }
$bins += @{ n = "rust Rc-boxed"; p = "$PSScriptRoot\rust\target\release\boxed.exe" }

foreach ($b in $bins) {
    $text = (& $b.p) -join " "
    $ok = $text -match "38837369|low digits: 369"
    $times = 1..$Runs | ForEach-Object {
        $sw = [Diagnostics.Stopwatch]::StartNew(); & $b.p | Out-Null; $sw.Stop(); $sw.Elapsed.TotalMilliseconds
    } | Sort-Object
    "{0,-22} min {1,8:N1} ms   median {2,8:N1} ms   max {3,8:N1} ms   checksum {4}" -f `
        $b.n, $times[0], $times[[int][math]::Floor($Runs / 2)], $times[-1], $(if ($ok) { "ok" } else { "MISMATCH: $text" })
}
