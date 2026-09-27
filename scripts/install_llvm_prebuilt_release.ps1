# Installs a Release LLVM/Clang/LLD/MLIR prefix without building LLVM, Clang or LLD.
#
# The official LLVM Windows package (clang+llvm-<ver>-x86_64-pc-windows-msvc.tar.xz, a GitHub
# release asset) carries the LLVM, Clang and LLD libraries with their CMake packages, lld/wasm-ld,
# clang and clang's resource directory, built with the static CRT (/MT) the project needs. It has
# no MLIR at all, so MLIR is built standalone against it (mlir/ from the matching source release)
# and installed into the same prefix. The result has the layout of a prefix made by
# config_llvm_release_vs.bat + build_llvm_release_vs.bat, so nothing that looks under
# 3rdParty/llvm/x64/release has to change.
#
# Differences from the custom build: RTTI and EH are off in the LLVM libraries (the project's MSVC
# flags force /GR /EHsc for its own code, which is fine), LLVMSupport brings rpmalloc as the
# process malloc (-INCLUDE:malloc on its CMake target), and there is no Debug variant - Debug still
# needs the custom build.
#
# Skips everything when the prefix already has MLIR, so it never overwrites a custom build.
param(
    [string]$Version = "22.1.8",
    [string]$Prefix = "",
    [string]$WorkDir = "",
    [string]$Generator = "Visual Studio 18 2026",
    [int]$Jobs = [Environment]::ProcessorCount
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
if ($Prefix -eq "") { $Prefix = Join-Path $repoRoot "3rdParty\llvm\x64\release" }
# Short by default: MSBuild's file tracker fails on paths past MAX_PATH, and MLIR's build tree is deep.
if ($WorkDir -eq "") { $WorkDir = Join-Path $repoRoot "__build\llvm-prebuilt" }
$Prefix = [IO.Path]::GetFullPath($Prefix)
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
# The CMake packages want forward slashes.
$prefixCMake = $Prefix -replace '\\', '/'

# Windows' own bsdtar reads .tar.xz; a Git for Windows tar earlier on PATH would need xz beside it.
$tar = Join-Path $env:SystemRoot "System32\tar.exe"

function Invoke-Checked([string]$what, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Get-Asset([string]$name) {
    $file = Join-Path $WorkDir $name
    if (-not (Test-Path $file)) {
        Write-Host "Downloading $name"
        Invoke-WebRequest -Uri "https://github.com/llvm/llvm-project/releases/download/llvmorg-$Version/$name" -OutFile "$file.part"
        Move-Item "$file.part" $file
    }
    return $file
}

$mlirConfig = Join-Path $Prefix "lib\cmake\mlir\MLIRConfig.cmake"
if (Test-Path $mlirConfig) {
    Write-Host "MLIR is already installed in $Prefix, nothing to do"
    exit 0
}

New-Item -ItemType Directory -Force -Path $WorkDir, $Prefix | Out-Null

# 1. The official package, unpacked into the prefix.
$llvmConfig = Join-Path $Prefix "lib\cmake\llvm\LLVMConfig.cmake"
if (-not (Test-Path $llvmConfig)) {
    $package = Get-Asset "clang+llvm-$Version-x86_64-pc-windows-msvc.tar.xz"
    Write-Host "Unpacking $package into $Prefix"
    Invoke-Checked "Unpacking the LLVM package" { & $tar -xf $package -C $Prefix --strip-components=1 }
}
if (-not (Select-String -Path $llvmConfig -SimpleMatch "set(LLVM_PACKAGE_VERSION $Version)" -Quiet)) {
    throw "$Prefix holds an LLVM other than $Version (see $llvmConfig)"
}

# 2. LLVMDebugInfoPDB's exported link line names the DIA SDK at the path it had on the machine
#    that built the package (llvm/llvm-project#86250), so every link that pulls it in fails with
#    LNK1181. Point it at this machine's Visual Studio.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -property installationPath
$diaGuids = Join-Path $vsPath "DIA SDK\lib\amd64\diaguids.lib"
if (-not (Test-Path $diaGuids)) { throw "DIA SDK not found at $diaGuids" }
$exports = Join-Path $Prefix "lib\cmake\llvm\LLVMExports.cmake"
$text = [IO.File]::ReadAllText($exports)
$fixed = [regex]::Replace($text, '[^";]*/DIA SDK/lib/amd64/diaguids\.lib', ($diaGuids -replace '\\', '/'))
if ($fixed -ne $text) {
    [IO.File]::WriteAllText($exports, $fixed)
    Write-Host "LLVMExports.cmake: DIA SDK -> $diaGuids"
}

# 3. MLIR sources: mlir/ plus the shared cmake/ modules it includes.
$srcRoot = Join-Path $WorkDir "llvm-project-$Version.src"
if (-not (Test-Path (Join-Path $srcRoot "mlir\CMakeLists.txt"))) {
    $sources = Get-Asset "llvm-project-$Version.src.tar.xz"
    Write-Host "Unpacking mlir/ and cmake/ from $sources"
    Invoke-Checked "Unpacking the MLIR sources" {
        & $tar -xf $sources -C $WorkDir "llvm-project-$Version.src/mlir" "llvm-project-$Version.src/cmake"
    }
}

# 4. MLIR, standalone against the unpacked LLVM, installed into the same prefix. LLVM_BUILD_UTILS
#    is off in the package's LLVMConfig, and without it mlir-tblgen is built but not installed.
$buildDir = Join-Path $WorkDir "mlir-build"
Invoke-Checked "Configuring MLIR" {
    cmake -S (Join-Path $srcRoot "mlir") -B $buildDir -G $Generator -A x64 -Thost=x64 -Wno-dev `
        "-DLLVM_DIR=$prefixCMake/lib/cmake/llvm" `
        "-DCMAKE_INSTALL_PREFIX=$prefixCMake" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
        -DLLVM_BUILD_UTILS=ON `
        -DLLVM_INCLUDE_TESTS=OFF `
        -DMLIR_INCLUDE_TESTS=OFF `
        -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF `
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
}
Invoke-Checked "Building MLIR" { cmake --build $buildDir --config Release --target install -j $Jobs }

foreach ($required in @($mlirConfig, (Join-Path $Prefix "bin\mlir-tblgen.exe"))) {
    if (-not (Test-Path $required)) { throw "MLIR install is missing $required" }
}
Write-Host "LLVM $Version with MLIR installed in $Prefix"
