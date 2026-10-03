# Shared by build_mlir_prebuilt_release.ps1 and install_llvm_prebuilt_release.ps1 (dot-sourced).
#
# The official LLVM Windows package (clang+llvm-<ver>-x86_64-pc-windows-msvc.tar.zst, a GitHub
# release asset of llvm/llvm-project) carries the LLVM, Clang and LLD libraries with their CMake
# packages, lld/wasm-ld, clang and clang's resource directory, built with the static CRT (/MT) the
# project needs. It has no MLIR at all. build_mlir_prebuilt_release.ps1 builds MLIR standalone
# against it once and packs the MLIR install; install_llvm_prebuilt_release.ps1 unpacks both into
# one prefix, which then has the layout of a prefix made by config_llvm_release_vs.bat +
# build_llvm_release_vs.bat. Nothing is compiled on the machine that installs.
#
# Differences from the custom build: RTTI and EH are off in the LLVM libraries (the project's MSVC
# flags force /GR /EHsc for its own code, which is fine), LLVMSupport brings rpmalloc as the
# process malloc (-INCLUDE:malloc on its CMake target), and there is no Debug variant - Debug still
# needs the custom build.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Windows' own bsdtar reads and writes .tar.zst; a Git for Windows tar earlier on PATH would need zst
# beside it.
$script:tar = Join-Path $env:SystemRoot "System32\tar.exe"

function Invoke-Checked([string]$what, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# Downloads $url into $dir unless it is already there; returns the file.
function Get-Download([string]$url, [string]$dir) {
    $file = Join-Path $dir ([IO.Path]::GetFileName(([Uri]$url).AbsolutePath))
    if (-not (Test-Path $file)) {
        Write-Host "Downloading $url"
        Invoke-WebRequest -Uri $url -OutFile "$file.part"
        Move-Item "$file.part" $file
    }
    return $file
}

function Get-LlvmAsset([string]$version, [string]$name, [string]$dir) {
    return Get-Download "https://github.com/llvm/llvm-project/releases/download/llvmorg-$version/$name" $dir
}

# Unpacks the official package into $prefix (unless it is there) and checks its version.
function Install-OfficialLlvm([string]$version, [string]$prefix, [string]$workDir) {
    $llvmConfig = Join-Path $prefix "lib\cmake\llvm\LLVMConfig.cmake"
    if (-not (Test-Path $llvmConfig)) {
        $package = Get-LlvmAsset $version "clang+llvm-$version-x86_64-pc-windows-msvc.tar.zst" $workDir
        New-Item -ItemType Directory -Force -Path $prefix | Out-Null
        Write-Host "Unpacking $package into $prefix"
        Invoke-Checked "Unpacking the LLVM package" { & $script:tar -xf $package -C $prefix --strip-components=1 }
    }
    if (-not (Select-String -Path $llvmConfig -SimpleMatch "set(LLVM_PACKAGE_VERSION $version)" -Quiet)) {
        throw "$prefix holds an LLVM other than $version (see $llvmConfig)"
    }
    Repair-DiaSdkPath $prefix
}

# LLVMDebugInfoPDB's exported link line names the DIA SDK at the path it had on the machine that
# built the package (llvm/llvm-project#86250), so every link that pulls it in fails with LNK1181.
# Point it at this machine's Visual Studio.
function Repair-DiaSdkPath([string]$prefix) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = & $vswhere -latest -products * -property installationPath
    $diaGuids = Join-Path $vsPath "DIA SDK\lib\amd64\diaguids.lib"
    if (-not (Test-Path $diaGuids)) { throw "DIA SDK not found at $diaGuids" }
    $exports = Join-Path $prefix "lib\cmake\llvm\LLVMExports.cmake"
    $text = [IO.File]::ReadAllText($exports)
    $fixed = [regex]::Replace($text, '[^";]*/DIA SDK/lib/amd64/diaguids\.lib', ($diaGuids -replace '\\', '/'))
    if ($fixed -ne $text) {
        [IO.File]::WriteAllText($exports, $fixed)
        Write-Host "LLVMExports.cmake: DIA SDK -> $diaGuids"
    }
}

# The MLIR archive's name, and where install_llvm_prebuilt_release.ps1 looks for it by default.
function Get-MlirArchiveName([string]$version) { return "mlir-$version-x86_64-pc-windows-msvc-mt.tar.xz" }
function Get-MlirArchiveUrl([string]$version) {
    return "https://github.com/ASDAlexander77/TypeScriptCompiler/releases/download/llvm-$version-mlir/$(Get-MlirArchiveName $version)"
}
