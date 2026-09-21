#!/usr/bin/env bash
# Phase 2 gate: programs built for 32-bit Windows (i686-pc-windows-msvc) link against the x86
# collector and runtime, and run correctly as 32-bit executables under every memory model.
#
# Each program in test/x86 is built with --opt (release: release CRT, release x86 libraries) and
# --no-default-lib for -mm=gc, -mm=rc and -mm=none. Each case checks that the executable's PE
# machine is IMAGE_FILE_MACHINE_I386 (0x014c), that it exits 0 and that stdout is exactly the
# expected text.
#
# gc_stress.ts keeps a 1000-node list alive while it allocates about a million garbage nodes plus
# strings and arrays; under -mm=gc that forces many collections, and the list's sum is wrong (or
# the program crashes) if a collection freed any of it.
set -u
TSLANG="${1:?usage: check-x86-run.sh <path to tslang.exe>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PROGRAMS="$REPO/tslang/test/x86"
READOBJ="$REPO/3rdParty/llvm/x64/release/bin/llvm-readobj.exe"

# Hermetic: the compiler falls back to these when a flag is missing, and they may name stale builds.
unset GC_LIB_PATH GC_SHARED_LIB_PATH TSLANG_LIB_PATH DEFAULT_LIB_PATH LLVM_LIB_PATH

# The documented layout: build_gc_release_vs_x86.bat copies the x86 gc.lib into the x86
# subdirectory of the x64 lib directory, so the flag x64 programs use serves x86 as well.
GC_DIR="$REPO/3rdParty/gc/x64/release/lib"
GC_X86="$GC_DIR/x86/gc.lib"
RT_DIR="$REPO/__build/tslang-runtime/release"
RT_X86="$RT_DIR/x86/TypeScriptAsyncRuntime.lib"

for f in "$TSLANG" "$READOBJ" "$GC_X86" "$RT_X86" "$PROGRAMS/hello.ts" "$PROGRAMS/gc_stress.ts"; do
    if [ ! -f "$f" ]; then
        echo "FAIL missing prerequisite: $f"
        exit 1
    fi
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Both flags name a directory whose x86 subdirectory holds the library, as the build scripts
# leave them: nothing is staged by hand.
GC_FLAG="--gc-lib-path=$GC_DIR"
RT_FLAG="--tslang-lib-path=$RT_DIR"

expected() {
    case "$1" in
        hello)     printf 'hello from 32-bit\n' ;;
        # sum(0..999) = 499500; sum of lengths of "round <r> of 200" for r in 0..199
        # = 200 * 13 + (10 * 1 + 90 * 2 + 100 * 3) = 3090
        gc_stress) printf '499500\n3090\n' ;;
    esac
}

fail=0
# run_case <program> <model> <extra tslang args>...
run_case() {
    local prog="$1" mm="$2"
    shift 2
    local name="$prog -mm=$mm" exe="$work/$prog-$mm.exe" err out status machine
    err="$("$TSLANG" --emit=exe --opt -mm="$mm" --no-default-lib -mtriple=i686-pc-windows-msvc \
        "$@" "$PROGRAMS/$prog.ts" -o "$exe" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -ne 0 ] || [ ! -f "$exe" ]; then
        echo "FAIL $name: compile failed (exit $status): $err"
        fail=1
        return
    fi
    machine="$("$READOBJ" --file-headers "$exe" | grep -m1 'Machine:')"
    if ! printf '%s' "$machine" | grep -q 'IMAGE_FILE_MACHINE_I386 (0x14C)'; then
        echo "FAIL $name: expected an I386 (0x14C) executable, got: $machine"
        fail=1
        return
    fi
    # The Windows CRT writes stdout in text mode, so lines end in CR LF.
    out="$("$exe" 2>"$work/stderr.txt" | tr -d '\r'; echo "exit=${PIPESTATUS[0]}")"
    if [ "$out" != "$(expected "$prog"; echo "exit=0")" ]; then
        echo "FAIL $name: expected output"
        expected "$prog" | sed 's/^/    /'
        echo "    exit=0"
        echo "  got"
        printf '%s\n' "$out" | sed 's/^/    /'
        sed 's/^/    stderr: /' "$work/stderr.txt"
        fail=1
        return
    fi
    echo "ok   $name"
}

for prog in hello gc_stress; do
    run_case "$prog" gc   "$GC_FLAG" "$RT_FLAG"
    run_case "$prog" rc   "$RT_FLAG"
    run_case "$prog" none "$RT_FLAG"
done

exit "$fail"
