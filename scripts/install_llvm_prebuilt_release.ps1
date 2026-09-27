# Installs a Release LLVM/Clang/LLD/MLIR prefix from prebuilt archives only - nothing is compiled:
# the official LLVM Windows package, then the MLIR archive made by build_mlir_prebuilt_release.ps1,
# unpacked into the same prefix. Background in llvm_prebuilt_common.ps1.
#
# Skips everything when the prefix already has MLIR, so it never overwrites a custom build.
param(
    [string]$Version = "22.1.8",
    [string]$Prefix = "",
    [string]$WorkDir = "",
    # A URL or a local file; defaults to the release asset named in llvm_prebuilt_common.ps1.
    [string]$MlirArchive = ""
)

. (Join-Path $PSScriptRoot "llvm_prebuilt_common.ps1")

$repoRoot = Split-Path -Parent $PSScriptRoot
if ($Prefix -eq "") { $Prefix = Join-Path $repoRoot "3rdParty\llvm\x64\release" }
if ($WorkDir -eq "") { $WorkDir = Join-Path $repoRoot "__build\llvm-prebuilt" }
if ($MlirArchive -eq "") { $MlirArchive = Get-MlirArchiveUrl $Version }
$Prefix = [IO.Path]::GetFullPath($Prefix)
$WorkDir = [IO.Path]::GetFullPath($WorkDir)

$mlirConfig = Join-Path $Prefix "lib\cmake\mlir\MLIRConfig.cmake"
if (Test-Path $mlirConfig) {
    Write-Host "MLIR is already installed in $Prefix, nothing to do"
    exit 0
}
New-Item -ItemType Directory -Force -Path $WorkDir, $Prefix | Out-Null

Install-OfficialLlvm $Version $Prefix $WorkDir

if ($MlirArchive -match '^https?://') { $MlirArchive = Get-Download $MlirArchive $WorkDir }
Write-Host "Unpacking $MlirArchive into $Prefix"
Invoke-Checked "Unpacking the MLIR archive" { & $script:tar -xf $MlirArchive -C $Prefix }

foreach ($required in @($mlirConfig, (Join-Path $Prefix "bin\mlir-tblgen.exe"))) {
    if (-not (Test-Path $required)) { throw "MLIR install is missing $required" }
}
Write-Host "LLVM $Version with MLIR installed in $Prefix"
