# Compile time against program size, for a set of program shapes.
#
# Generates each shape at each size (one .ts file of n units), compiles it to an object file and
# prints the wall time (best of two) and how it grows from one size to the next. A shape that grows
# much more than the size does (2x for twice the source) has something quadratic in it.
#
#   pwsh tslang/benchmarks/compile-time/shapes.ps1 [-Tslang <tslang.exe>] [-Sizes 500,1000,2000]
#        [-Only generics,interfaces] [-Flags --opt,--opt_level=3] [-Stages]
#
# -Stages prints, instead, the per-stage times of the two largest sizes from --mlir-timing: which
# stage of the compiler the growth is in.
param(
    [string]$Tslang = "$PSScriptRoot\..\..\..\__build\tslang\windows-msbuild-2026-release\bin\tslang.exe",
    [string[]]$Sizes = @('500', '1000', '2000'),
    [string[]]$Only = @(),
    [string[]]$Flags = @(),
    [switch]$Stages
)

$ErrorActionPreference = "Continue"
# `pwsh -File shapes.ps1 -Only a,b` hands the list over as one string
$Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Flags = @($Flags | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
[int[]]$Sizes = @($Sizes | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$out = Join-Path $PSScriptRoot "out"
New-Item -ItemType Directory -Force $out | Out-Null

function Lines([scriptblock]$body) {
    $sb = [Text.StringBuilder]::new()
    & $body $sb
    return $sb.ToString()
}

$shapes = [ordered]@{
    'functions' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) {
            [void]$sb.AppendLine("function f$i(a: number, b: number): number {")
            [void]$sb.AppendLine("    let s = 0;")
            [void]$sb.AppendLine("    for (let i = 0; i < a; i++) { if (i % 3 == 0) s += i * b; else s -= b; }")
            [void]$sb.AppendLine("    return s + a;")
            [void]$sb.AppendLine("}")
        }
        [void]$sb.AppendLine("function main() { print(f0(1, 2)); }") } }
    'classes' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) {
            [void]$sb.AppendLine("class C$i {")
            [void]$sb.AppendLine("    x: number; y: number;")
            [void]$sb.AppendLine("    constructor(x: number, y: number) { this.x = x; this.y = y; }")
            [void]$sb.AppendLine("    len(): number { return this.x * this.x + this.y * this.y; }")
            [void]$sb.AppendLine("}")
        }
        [void]$sb.AppendLine("function main() { print(1); }") } }
    'strings' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("function s$i(): string { return `"text number $i of the program`"; }") }
        [void]$sb.AppendLine("function main() { print(s0()); }") } }
    'interfaces' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) {
            [void]$sb.AppendLine("interface I$i { area(): number; }")
            [void]$sb.AppendLine("class K$i implements I$i { w: number = $i; area(): number { return this.w * 2; } }")
            [void]$sb.AppendLine("function use$i(): number { const v: I$i = new K$i(); return v.area(); }")
        }
        [void]$sb.AppendLine("function main() { print(use0()); }") } }
    'inheritance' = { param($n) Lines { param($sb)
        [void]$sb.AppendLine("class B { v: number = 1; get(): number { return this.v; } }")
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("class D$i extends B { get(): number { return this.v + $i; } }") }
        [void]$sb.AppendLine("function main() { const b: B = new D0(); print(b.get()); }") } }
    'generics' = { param($n) Lines { param($sb)
        [void]$sb.AppendLine("class Box<T> { value: T; constructor(value: T) { this.value = value; } get(): T { return this.value; } }")
        for ($i = 0; $i -lt $n; $i++) {
            [void]$sb.AppendLine("class P$i { n: number = $i; }")
            [void]$sb.AppendLine("function g$i(): number { const b = new Box<P$i>(new P$i()); return b.get().n; }")
        }
        [void]$sb.AppendLine("function main() { print(g0()); }") } }
    'globals' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("let v$i = $i * 2 + 1;") }
        [void]$sb.AppendLine("function main() { print(v0 + v$($n - 1)); }") } }
    'big function' = { param($n) Lines { param($sb)
        [void]$sb.AppendLine("function big(a: number): number {"); [void]$sb.AppendLine("    let s = a;")
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("    if (s > $i) { s = s + $i; } else { s = s * 2 - $i; }") }
        [void]$sb.AppendLine("    return s; }"); [void]$sb.AppendLine("function main() { print(big(3)); }") } }
    'closures' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("function c$i(a: number): number { const f = (x: number) => x + a + $i; return f(1); }") }
        [void]$sb.AppendLine("function main() { print(c0(1)); }") } }
    'object literals' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("function o$i(): number { const p = { a$i`: $i, b: 2 }; return p.a$i + p.b; }") }
        [void]$sb.AppendLine("function main() { print(o0()); }") } }
    'arrays' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("function a$i(): number { const xs = [$i, 2, 3, 4]; let s = 0; for (const x of xs) s += x; return s; }") }
        [void]$sb.AppendLine("function main() { print(a0()); }") } }
    'calls' = { param($n) Lines { param($sb)
        [void]$sb.AppendLine("function f0(a: number): number { return a + 1; }")
        for ($i = 1; $i -lt $n; $i++) { [void]$sb.AppendLine("function f$i(a: number): number { return f$($i - 1)(a) + f$([int]($i / 2))(a); }") }
        [void]$sb.AppendLine("function main() { print(f5(1)); }") } }
    'try/catch' = { param($n) Lines { param($sb)
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("function t$i(a: number): number { try { if (a > $i) throw $i; return a; } catch (e: number) { return e; } }") }
        [void]$sb.AppendLine("function main() { print(t0(1)); }") } }
    'enums+switch' = { param($n) Lines { param($sb)
        [void]$sb.Append("enum E {"); for ($i = 0; $i -lt $n; $i++) { [void]$sb.Append(" M$i,") }; [void]$sb.AppendLine(" }")
        [void]$sb.AppendLine("function sw(e: E): number { switch (e) {")
        for ($i = 0; $i -lt $n; $i++) { [void]$sb.AppendLine("    case E.M$i`: return $i;") }
        [void]$sb.AppendLine("    } return -1; }"); [void]$sb.AppendLine("function main() { print(sw(E.M1)); }") } }
}

function SourceOf([string]$name, [int]$n) {
    $src = Join-Path $out "$($name -replace '[^a-z]', '_')_$n.ts"
    Set-Content $src (& $shapes[$name] $n)
    return $src
}

function StagesOf([string]$src) {
    $r = [ordered]@{}
    $lines = & $Tslang --emit=obj --no-default-lib --mlir-timing @Flags $src -o "$out\x.obj" 2>&1 | ForEach-Object { "$_" }
    foreach ($l in $lines) {
        if ($l -match '^\s*[\d.]+ \(\s*[\d.]+%\)\s+([\d.]+) \(\s*[\d.]+%\)\s+(.+?)\s*$') {
            $stage = $Matches[2] -replace "``anonymous-namespace'::", ''
            $r[$stage] = [double]$Matches[1] + $(if ($r.Contains($stage)) { $r[$stage] } else { 0 })
        }
    }
    return $r
}

"flags: --emit=obj --no-default-lib $($Flags -join ' ')"
if ($Stages) {
    $small = $Sizes[-2]; $large = $Sizes[-1]
    foreach ($name in $shapes.Keys) {
        if ($Only.Count -and $Only -notcontains $name) { continue }
        $a = StagesOf (SourceOf $name $small); $b = StagesOf (SourceOf $name $large)
        "===== $name   n=$small -> n=$large"
        $b.Keys | Sort-Object { -$b[$_] } | Select-Object -First 10 | ForEach-Object {
            $x = if ($a.Contains($_)) { $a[$_] } else { 0 }
            "{0,8:N3} -> {1,8:N3}  {2,5}  {3}" -f $x, $b[$_], $(if ($x -gt 0.004) { "{0:N1}x" -f ($b[$_] / $x) } else { '' }), $_
        }
    }
    return
}

$header = "{0,-16}" -f 'shape'; foreach ($n in $Sizes) { $header += "{0,10}" -f "n=$n" }
$header + "   growth"
foreach ($name in $shapes.Keys) {
    if ($Only.Count -and $Only -notcontains $name) { continue }
    $row = "{0,-16}" -f $name; $times = @(); $failed = $false
    foreach ($n in $Sizes) {
        $src = SourceOf $name $n
        $best = [double]::MaxValue
        foreach ($i in 1..2) {
            $sw = [Diagnostics.Stopwatch]::StartNew(); & $Tslang --emit=obj --no-default-lib @Flags $src -o "$out\x.obj" *> "$out\last.log"; $sw.Stop()
            if ($LASTEXITCODE -ne 0) { $failed = $true; break }
            $best = [Math]::Min($best, $sw.Elapsed.TotalSeconds)
            if ($best -gt 60) { break }
        }
        if ($failed) { $row += "{0,10}" -f 'FAIL'; $row += "   " + ((Get-Content "$out\last.log" -TotalCount 2) -join ' | '); break }
        $times += $best; $row += "{0,10:N2}" -f $best
        if ($best -gt 60) { $row += '   (stopped: over a minute)'; break }
    }
    if (-not $failed -and $times.Count -ge 2) {
        $growth = @(); for ($k = 1; $k -lt $times.Count; $k++) { $growth += "{0:N1}x" -f ($times[$k] / $times[$k - 1]) }
        $row += "   " + ($growth -join ', ')
    }
    $row
}
