# Builds the prebuilt MLIR archive that install_llvm_prebuilt_release.ps1 installs - run once per
# LLVM version, by hand, then upload the archive as the asset of the release named in
# Get-MlirArchiveUrl (llvm_prebuilt_common.ps1). Background there.
#
# MLIR is built standalone (mlir/ and cmake/ from the matching llvm-project source release) against
# the official LLVM package, Release, static CRT, and installed into an empty prefix. That install
# is the archive: its CMake package finds LLVM relative to itself, so unpacked over the official
# package it needs no absolute paths. About 10 minutes at -j20; far longer on a 4-core CI runner,
# which is why CI downloads the result instead.
param(
    [string]$Version = "22.1.8",
    [string]$WorkDir = "",
    [string]$Out = "",
    [string]$Generator = "Visual Studio 18 2026",
    [int]$Jobs = [Environment]::ProcessorCount
)

. (Join-Path $PSScriptRoot "llvm_prebuilt_common.ps1")

$repoRoot = Split-Path -Parent $PSScriptRoot
# Short by default: MSBuild's file tracker fails on paths past MAX_PATH, and MLIR's build tree is deep.
if ($WorkDir -eq "") { $WorkDir = Join-Path $repoRoot "__build\llvm-prebuilt" }
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if ($Out -eq "") { $Out = Join-Path $WorkDir (Get-MlirArchiveName $Version) }
$Out = [IO.Path]::GetFullPath($Out)
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# 1. The official package to build against.
$llvmPrefix = Join-Path $WorkDir "llvm"
Install-OfficialLlvm $Version $llvmPrefix $WorkDir

# 2. MLIR sources: mlir/ plus the shared cmake/ modules it includes.
$srcRoot = Join-Path $WorkDir "llvm-project-$Version.src"
if (-not (Test-Path (Join-Path $srcRoot "mlir\CMakeLists.txt"))) {
    $sources = Get-LlvmAsset $Version "llvm-project-$Version.src.tar.xz" $WorkDir
    Write-Host "Unpacking mlir/ and cmake/ from $sources"
    Invoke-Checked "Unpacking the MLIR sources" {
        & $script:tar -xf $sources -C $WorkDir "llvm-project-$Version.src/mlir" "llvm-project-$Version.src/cmake"
    }
}

# 3. MLIR into an empty prefix. LLVM_BUILD_UTILS is off in the package's LLVMConfig, and without
#    it mlir-tblgen is built but not installed.
$buildDir = Join-Path $WorkDir "mlir-build"
$mlirPrefix = Join-Path $WorkDir "mlir-install"
if (Test-Path $mlirPrefix) { Remove-Item -Recurse -Force $mlirPrefix }
Invoke-Checked "Configuring MLIR" {
    cmake -S (Join-Path $srcRoot "mlir") -B $buildDir -G $Generator -A x64 -Thost=x64 -Wno-dev `
        "-DLLVM_DIR=$(($llvmPrefix -replace '\\', '/'))/lib/cmake/llvm" `
        "-DCMAKE_INSTALL_PREFIX=$(($mlirPrefix -replace '\\', '/'))" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
        -DLLVM_BUILD_UTILS=ON `
        -DLLVM_INCLUDE_TESTS=OFF `
        -DMLIR_INCLUDE_TESTS=OFF `
        -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF `
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
}
Invoke-Checked "Building MLIR" { cmake --build $buildDir --config Release --target install -j $Jobs }
foreach ($required in @("lib\cmake\mlir\MLIRConfig.cmake", "bin\mlir-tblgen.exe")) {
    if (-not (Test-Path (Join-Path $mlirPrefix $required))) { throw "MLIR install is missing $required" }
}

# 4. The archive.
if (Test-Path $Out) { Remove-Item -Force $Out }
Invoke-Checked "Packing MLIR" { & $script:tar -cJf $Out -C $mlirPrefix . }
Write-Host "Wrote $Out ($([math]::Round((Get-Item $Out).Length / 1MB)) MB)"
Write-Host "Upload it as $(Get-MlirArchiveUrl $Version)"
