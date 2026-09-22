#!/usr/bin/env bash
# Phase 2 gate: linking for x86 Windows takes the collector and the runtime from an `x86`
# subdirectory of the configured lib path, and refuses, with a message naming the library, its
# machine and the script that builds it, a missing x86 build or a library built for another
# machine. Without the check the linker reports LNK4272 and a page of unresolved symbols.
set -u
TSLANG="${1:?usage: check-x86-libs.sh <path to tslang.exe> [x64 TypeScriptAsyncRuntime.lib]}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# The x64 runtime is only copied into wrong-machine layouts; the in-tree build's is the default.
RT_X64="${2:-$REPO/__build/tslang/windows-msbuild-2026-release/lib/TypeScriptAsyncRuntime.lib}"

GC_X64="$REPO/3rdParty/gc/x64/release/lib/gc.lib"
GC_X86="$REPO/3rdParty/gc/x86/release/lib/gc.lib"
GCDLL_X86_LIB="$REPO/3rdParty/gcdll/x86/release/lib/gc.lib"
GCDLL_X86_DLL="$REPO/3rdParty/gcdll/x86/release/bin/gc.dll"
GCDLL_X64_DLL="$REPO/3rdParty/gcdll/x64/release/bin/gc.dll"
RT_X86="$REPO/__build/tslang-runtime/release/x86/TypeScriptAsyncRuntime.lib"
# The default-lib repo is a sibling checkout; when its x64 build exists, its release/gc static lib
# stands in for "a default library, but the wrong one" below and is the only thing the x64-control
# case can run against. It is optional, unlike the libraries in the prerequisite loop below: a
# machine with no TypeScriptCompilerDefaultLib x64 build still runs every other case in this file
# (the "x64 lib in x86 tree" case below needs only some x64 COFF archive, not a real default
# library); only the x64-control case is SKIPPED. There is no x86 default library staged here -
# the x86 default library (spec Phase 4, As built) has its own run coverage in check-x86-run.sh's
# default-library section - so this file has no positive x86 case.
DEFAULTLIB_X64_LIB="$REPO/../TypeScriptCompilerDefaultLib/__build/defaultlib/lib/release/gc/TypeScriptDefaultLib.lib"

for f in "$GC_X64" "$GC_X86" "$GCDLL_X86_LIB" "$GCDLL_X86_DLL" "$GCDLL_X64_DLL" "$RT_X64" "$RT_X86"; do
    if [ ! -f "$f" ]; then
        echo "FAIL missing prerequisite: $f"
        exit 1
    fi
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
printf 'print("x");\n' > "$work/x.ts"

# Layouts: <dir> is what --*-lib-path names; x86 libraries sit in <dir>/x86.
mkdir -p "$work/empty" \
         "$work/gc-good/x86" "$work/gc-x64-in-x86/x86" "$work/gc-x86-flat" \
         "$work/gcdll-good/x86" "$work/gcdll-x64-dll/x86" \
         "$work/rt-good/x86" "$work/rt-x64-in-x86/x86" \
         "$work/gcdll-x86-flat" "$work/rt-x86-flat" \
         "$work/defaultlib-x64-only/defaultlib/lib/release/gc" \
         "$work/defaultlib-x64-in-x86/defaultlib/lib/x86/release/gc"
cp "$GC_X86" "$work/gc-good/x86/gc.lib"
cp "$GC_X64" "$work/gc-x64-in-x86/x86/gc.lib"
cp "$GC_X86" "$work/gc-x86-flat/gc.lib"
cp "$GCDLL_X86_LIB" "$work/gcdll-good/x86/gc.lib"
cp "$GCDLL_X86_DLL" "$work/gcdll-good/x86/gc.dll"
cp "$GCDLL_X86_LIB" "$work/gcdll-x64-dll/x86/gc.lib"
cp "$GCDLL_X64_DLL" "$work/gcdll-x64-dll/x86/gc.dll"
cp "$RT_X86" "$work/rt-good/x86/TypeScriptAsyncRuntime.lib"
cp "$RT_X64" "$work/rt-x64-in-x86/x86/TypeScriptAsyncRuntime.lib"
cp "$GCDLL_X86_LIB" "$work/gcdll-x86-flat/gc.lib"
cp "$GCDLL_X86_DLL" "$work/gcdll-x86-flat/gc.dll"
cp "$RT_X86" "$work/rt-x86-flat/TypeScriptAsyncRuntime.lib"
# defaultlib-x64-only/defaultlib/lib/release/gc: a stand-in file. The "no x86 tree" case below
# never opens it - it fails on the missing x86 subdirectory before reading any file - and the
# "x64 control" case only runs against this tree, with the real x64 default library copied over
# it, when one is staged (see below). defaultlib-x64-in-x86 needs only *some* x64 COFF archive:
# the machine check reads the file's COFF header before ever trying to link it as a default
# library, so $GC_X64 stands in without this file depending on a TypeScriptCompilerDefaultLib build.
cp "$GC_X64" "$work/defaultlib-x64-only/defaultlib/lib/release/gc/TypeScriptDefaultLib.lib"
cp "$GC_X64" "$work/defaultlib-x64-in-x86/defaultlib/lib/x86/release/gc/TypeScriptDefaultLib.lib"

X86="i686-pc-windows-msvc"
X64="x86_64-pc-windows-msvc"

fail=0
# expect_error <name> <triple> <emit> <substring>... -- <extra tslang args>...
expect_error() {
    local name="$1" triple="$2" emit="$3"
    shift 3
    local needles=()
    while [ "$1" != "--" ]; do
        needles+=("$1")
        shift
    done
    shift
    local err status
    err="$("$TSLANG" --emit="$emit" --opt -mm=gc --no-default-lib -mtriple="$triple" "$@" \
        "$work/x.ts" -o "$work/out.$emit" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "FAIL $name: expected a non-zero exit, got 0"
        fail=1
        return
    fi
    for needle in "${needles[@]}"; do
        if ! printf '%s' "$err" | grep -q -F -- "$needle"; then
            echo "FAIL $name: expected the error to contain '$needle', got: $err"
            fail=1
            return
        fi
    done
    if printf '%s' "$err" | grep -q -F -- "LNK"; then
        echo "FAIL $name: the linker ran; expected tslang to stop first, got: $err"
        fail=1
        return
    fi
    echo "ok   $name"
}

# No flat-path complaint about a correctly staged x86 layout, however the link itself ends.
expect_no_flat_error() {
    local name="$1" emit="$2"
    shift 2
    local err
    err="$("$TSLANG" --emit="$emit" --opt -mm=gc --no-default-lib -mtriple="$X86" "$@" \
        "$work/x.ts" -o "$work/good.$emit" 2>&1 >/dev/null)"
    if printf '%s' "$err" | grep -q -e "is not pointing to file" -e "no x86 build" -e "but this program targets"; then
        echo "FAIL $name: unexpected library error: $err"
        fail=1
    else
        echo "ok   $name"
    fi
}

# The default library resolution below only runs without --no-default-lib, unlike every case
# above, so it is a separate pair of helpers rather than a reuse of expect_error/expect_no_flat_error.
# A distinct -o per call: unlike the --no-default-lib cases above, these can produce a real linked
# .exe, and Windows (AV scan) can briefly hold a lock on one just written, which would otherwise
# spuriously fail the next case's read of the shared work dir.
dl_case_n=0

# expect_default_lib_error <name> <triple> <default-lib-path> <substring>... -- <extra tslang args>...
expect_default_lib_error() {
    local name="$1" triple="$2" dlpath="$3"
    shift 3
    local needles=()
    while [ "$1" != "--" ]; do
        needles+=("$1")
        shift
    done
    shift
    dl_case_n=$((dl_case_n + 1))
    local err status
    err="$("$TSLANG" --emit=exe --opt -mm=gc -mtriple="$triple" --default-lib-path="$dlpath" "$@" \
        "$work/x.ts" -o "$work/out-dl-$dl_case_n.exe" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "FAIL $name: expected a non-zero exit, got 0"
        fail=1
        return
    fi
    for needle in "${needles[@]}"; do
        if ! printf '%s' "$err" | grep -q -F -- "$needle"; then
            echo "FAIL $name: expected the error to contain '$needle', got: $err"
            fail=1
            return
        fi
    done
    echo "ok   $name"
}

# expect_default_lib_ok <name> <triple> <default-lib-path> <extra tslang args>...
expect_default_lib_ok() {
    local name="$1" triple="$2" dlpath="$3"
    shift 3
    dl_case_n=$((dl_case_n + 1))
    local err status
    err="$("$TSLANG" --emit=exe --opt -mm=gc -mtriple="$triple" --default-lib-path="$dlpath" "$@" \
        "$work/x.ts" -o "$work/out-dl-$dl_case_n.exe" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "FAIL $name: expected a zero exit, got $status: $err"
        fail=1
    else
        echo "ok   $name"
    fi
}

GOOD_RT="--tslang-lib-path=$work/rt-good"
GOOD_GC="--gc-lib-path=$work/gc-good"

expect_error "gc: no x86 subdirectory"        "$X86" exe "x86" "build_gc_release_vs_x86" -- \
    "--gc-lib-path=$work/empty" "$GOOD_RT"
expect_error "gc: x64 gc.lib in x86/"         "$X86" exe "gc.lib" "x64" "x86" "but this program targets x86" -- \
    "--gc-lib-path=$work/gc-x64-in-x86" "$GOOD_RT"
expect_error "runtime: no x86 subdirectory"   "$X86" exe "x86" "build_tslang_runtime_release_x86" -- \
    "$GOOD_GC" "--tslang-lib-path=$work/empty"
expect_error "runtime: x64 runtime in x86/"   "$X86" exe "TypeScriptAsyncRuntime.lib" "x64" "x86" "but this program targets x86" -- \
    "$GOOD_GC" "--tslang-lib-path=$work/rt-x64-in-x86"
expect_error "gcdll: no x86 subdirectory"     "$X86" dll "x86" "build_gc_release_shared_vs_x86" -- \
    "--gc-shared-lib-path=$work/empty" "$GOOD_GC" "$GOOD_RT"
expect_error "gcdll: x64 gc.dll beside x86 gc.lib" "$X86" dll "gc.dll" "x64" "but this program targets x86" -- \
    "--gc-shared-lib-path=$work/gcdll-x64-dll" "$GOOD_GC" "$GOOD_RT"
expect_error "x64: x86 gc.lib in flat path"   "$X64" exe "gc.lib" "x86" "but this program targets x64" -- \
    "--gc-lib-path=$work/gc-x86-flat" "--tslang-lib-path=$(dirname "$RT_X64")"
expect_error "x64: x86 gc.dll import lib in flat path" "$X64" dll "gc.lib" "x86" "but this program targets x64" -- \
    "--gc-shared-lib-path=$work/gcdll-x86-flat" "--gc-lib-path=$(dirname "$GC_X64")" "--tslang-lib-path=$(dirname "$RT_X64")"
expect_error "x64: x86 runtime lib in flat path" "$X64" exe "TypeScriptAsyncRuntime.lib" "x86" "but this program targets x64" -- \
    "--gc-lib-path=$(dirname "$GC_X64")" "--tslang-lib-path=$work/rt-x86-flat"

expect_no_flat_error "exe: correct x86 layout" exe "$GOOD_GC" "$GOOD_RT"
expect_no_flat_error "dll: correct x86 layout" dll "--gc-shared-lib-path=$work/gcdll-good" "$GOOD_GC" "$GOOD_RT"

# Default library, without --no-default-lib: it takes the same x86-tree/machine-check treatment
# as the collector and the runtime above. There is no positive x86 case here (no x86 default
# library is staged for this script to find); check-x86-run.sh's default-library section adds run
# coverage once one is built.
expect_default_lib_error "default lib: no x86 tree" "$X86" "$work/defaultlib-x64-only" \
    "no x86 default library built for -mm=gc" "defaultlib" "x86" "release" "gc" -- \
    "$GOOD_GC" "$GOOD_RT"
expect_default_lib_error "default lib: x64 lib in x86 tree" "$X86" "$work/defaultlib-x64-in-x86" \
    "is built for x64" -- \
    "$GOOD_GC" "$GOOD_RT"
# Only this case needs a real, linkable x64 default library (an actual TypeScriptCompilerDefaultLib
# build); the two error cases above are satisfied by the stand-in files staged above. SKIPPED, not
# FAILed, when none is staged, matching check-x86-run.sh's skip style for the same situation.
if [ -f "$DEFAULTLIB_X64_LIB" ]; then
    cp "$DEFAULTLIB_X64_LIB" "$work/defaultlib-x64-only/defaultlib/lib/release/gc/TypeScriptDefaultLib.lib"
    expect_default_lib_ok "default lib: x64 control" "$X64" "$work/defaultlib-x64-only" \
        "--gc-lib-path=$(dirname "$GC_X64")" "--tslang-lib-path=$(dirname "$RT_X64")"
else
    echo "SKIPPED: default lib: x64 control (no x64 default library built at $DEFAULTLIB_X64_LIB;" \
         "build it in TypeScriptCompilerDefaultLib)"
fi

exit "$fail"
