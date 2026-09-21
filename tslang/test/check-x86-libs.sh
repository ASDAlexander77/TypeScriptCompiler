#!/usr/bin/env bash
# Phase 2 gate: linking for x86 Windows takes the collector and the runtime from an `x86`
# subdirectory of the configured lib path, and refuses, with a message naming the library, its
# machine and the script that builds it, a missing x86 build or a library built for another
# machine. Without the check the linker reports LNK4272 and a page of unresolved symbols.
set -u
TSLANG="${1:?usage: check-x86-libs.sh <path to tslang.exe>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

GC_X64="$REPO/3rdParty/gc/x64/release/lib/gc.lib"
GC_X86="$REPO/3rdParty/gc/x86/release/lib/gc.lib"
GCDLL_X86_LIB="$REPO/3rdParty/gcdll/x86/release/lib/gc.lib"
GCDLL_X86_DLL="$REPO/3rdParty/gcdll/x86/release/bin/gc.dll"
RT_X64="$REPO/__build/tslang-runtime/release/TypeScriptAsyncRuntime.lib"
RT_X86="$REPO/__build/tslang-runtime/release/x86/TypeScriptAsyncRuntime.lib"

for f in "$GC_X64" "$GC_X86" "$GCDLL_X86_LIB" "$GCDLL_X86_DLL" "$RT_X64" "$RT_X86"; do
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
         "$work/gcdll-good/x86" \
         "$work/rt-good/x86" "$work/rt-x64-in-x86/x86" \
         "$work/gcdll-x86-flat" "$work/rt-x86-flat"
cp "$GC_X86" "$work/gc-good/x86/gc.lib"
cp "$GC_X64" "$work/gc-x64-in-x86/x86/gc.lib"
cp "$GC_X86" "$work/gc-x86-flat/gc.lib"
cp "$GCDLL_X86_LIB" "$work/gcdll-good/x86/gc.lib"
cp "$GCDLL_X86_DLL" "$work/gcdll-good/x86/gc.dll"
cp "$RT_X86" "$work/rt-good/x86/TypeScriptAsyncRuntime.lib"
cp "$RT_X64" "$work/rt-x64-in-x86/x86/TypeScriptAsyncRuntime.lib"
cp "$GCDLL_X86_LIB" "$work/gcdll-x86-flat/gc.lib"
cp "$GCDLL_X86_DLL" "$work/gcdll-x86-flat/gc.dll"
cp "$RT_X86" "$work/rt-x86-flat/TypeScriptAsyncRuntime.lib"

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
expect_error "x64: x86 gc.lib in flat path"   "$X64" exe "gc.lib" "x86" "but this program targets x64" -- \
    "--gc-lib-path=$work/gc-x86-flat" "--tslang-lib-path=$(dirname "$RT_X64")"
expect_error "x64: x86 gc.dll import lib in flat path" "$X64" dll "gc.lib" "x86" "but this program targets x64" -- \
    "--gc-shared-lib-path=$work/gcdll-x86-flat" "--gc-lib-path=$(dirname "$GC_X64")" "--tslang-lib-path=$(dirname "$RT_X64")"
expect_error "x64: x86 runtime lib in flat path" "$X64" exe "TypeScriptAsyncRuntime.lib" "x86" "but this program targets x64" -- \
    "--gc-lib-path=$(dirname "$GC_X64")" "--tslang-lib-path=$work/rt-x86-flat"

expect_no_flat_error "exe: correct x86 layout" exe "$GOOD_GC" "$GOOD_RT"
expect_no_flat_error "dll: correct x86 layout" dll "--gc-shared-lib-path=$work/gcdll-good" "$GOOD_GC" "$GOOD_RT"

exit "$fail"
