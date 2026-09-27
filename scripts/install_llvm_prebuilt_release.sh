#!/bin/sh
# Installs a Release LLVM/Clang/LLD/MLIR prefix from the official LLVM Linux package
# (LLVM-<ver>-Linux-X64.tar.xz, a GitHub release asset of llvm/llvm-project) - nothing is compiled.
# Unlike the Windows package it has MLIR, so it is all that is needed; it unpacks into the prefix
# config_llvm_release.sh + build_llvm_release.sh would make, so nothing that looks under
# 3rdParty/llvm/release has to change. Windows: install_llvm_prebuilt_release.ps1.
#
# Differences from the custom build: RTTI and EH are off in the LLVM libraries (config_macros.cmake
# then builds the project with -fno-rtti), and LLVMSupport needs zlib and zstd (zlib1g-dev,
# libzstd-dev) to link. There is no Debug variant - Debug still needs the custom build.
#
# The package is about 12 GB unpacked, mostly tools. Left out: flang, lldb and bolt, which the LLVM,
# Clang and MLIR CMake packages do not reference (every file those do reference must exist, or
# find_package fails). The rest stays, about 9.6 GB.
#
# Usage: install_llvm_prebuilt_release.sh [version]   (PREFIX / WORK_DIR override the defaults)
# Skips everything when the prefix already has MLIR, so it never overwrites a custom build.
set -e

VERSION=${1:-22.1.8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${PREFIX:-$ROOT/3rdParty/llvm/release}
WORK_DIR=${WORK_DIR:-$ROOT/__build/llvm-prebuilt}

if [ -f "$PREFIX/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "MLIR is already installed in $PREFIX, nothing to do"
    exit 0
fi

NAME=LLVM-$VERSION-Linux-X64
PACKAGE=$WORK_DIR/$NAME.tar.xz
mkdir -p "$WORK_DIR" "$PREFIX"
if [ ! -f "$PACKAGE" ]; then
    echo "Downloading $NAME.tar.xz"
    curl -fsSL --retry 3 -o "$PACKAGE.part" \
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-$VERSION/$NAME.tar.xz"
    mv "$PACKAGE.part" "$PACKAGE"
fi

echo "Unpacking $PACKAGE into $PREFIX"
xz -T0 -dc "$PACKAGE" | tar -x -C "$PREFIX" --strip-components=1 \
    --exclude='*/bin/flang*' --exclude='*/bin/bbc' --exclude='*/bin/tco' \
    --exclude='*/bin/fir-*' --exclude='*/bin/f18-*' --exclude='*/bin/lldb*' \
    --exclude='*/bin/llvm-bolt*' --exclude='*/bin/*bolt*' --exclude='*/bin/merge-fdata' \
    --exclude='*/lib/liblldb*' --exclude='*/lib/libFortran*' --exclude='*/lib/libflang*' \
    --exclude='*/lib/libFlang*' --exclude='*/lib/libFIR*' --exclude='*/lib/libHLFIR*' \
    --exclude='*/lib/libCUF*' --exclude='*/lib/libMIF*' --exclude='*/lib/libbolt*' \
    --exclude='*/lib/libLLVMBOLT*' --exclude='*/lib/python*' \
    --exclude='*/include/flang' --exclude='*/include/lldb' --exclude='*/include/bolt'
# The download is 1.9 GB; the prefix is what the build needs.
rm -f "$PACKAGE"

if ! grep -q "set(LLVM_PACKAGE_VERSION $VERSION)" "$PREFIX/lib/cmake/llvm/LLVMConfig.cmake"; then
    echo "$PREFIX holds an LLVM other than $VERSION" >&2
    exit 1
fi
for required in lib/cmake/mlir/MLIRConfig.cmake lib/cmake/clang/ClangConfig.cmake bin/mlir-tblgen bin/ld.lld; do
    if [ ! -e "$PREFIX/$required" ]; then
        echo "The LLVM package is missing $required" >&2
        exit 1
    fi
done
echo "LLVM $VERSION with MLIR installed in $PREFIX"
