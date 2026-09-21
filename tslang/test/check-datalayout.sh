#!/usr/bin/env bash
# Phase 1 gate: emitted IR must name a data layout that matches its triple, so that --emit=llvm
# output is self-describing rather than carrying LLVM's default layout under a 32-bit triple.
set -u
TSLANG="${1:?usage: check-datalayout.sh <path to tslang.exe>}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
printf 'print("x");\n' > "$work/dl.ts"

fail=0
check() {
    local triple="$1" expect="$2"
    "$TSLANG" --emit=llvm -mm=none --no-default-lib -mtriple="$triple" "$work/dl.ts" \
        -o "$work/dl.ll" >/dev/null 2>&1
    local got
    got="$(grep -m1 '^target datalayout' "$work/dl.ll" || true)"
    if [ -z "$got" ]; then
        echo "FAIL $triple: no 'target datalayout' in emitted IR"
        fail=1
    elif ! printf '%s' "$got" | grep -q -- "$expect"; then
        echo "FAIL $triple: expected a layout containing '$expect', got: $got"
        fail=1
    else
        echo "ok   $triple"
    fi
}

# p:32: and p:64: are the pointer-size entries; they are what this phase is about.
check "i686-pc-windows-msvc"    "p:32:"
check "wasm32-unknown-unknown"  "p:32:"
check "x86_64-pc-windows-msvc"  "-p270:" # x86-64 layouts carry the AS270/271/272 entries
# wasm32's layout used to be a hardcoded string with f128:64 and no i128:128, matching neither of
# LLVM's own wasm32 derivations; lowering now reads the TargetMachine's layout from the module.
check "wasm32-unknown-unknown"  "i128:128"

# A triple the target registry cannot resolve must be a hard, diagnosed failure rather than
# silently emitted IR that names the triple but carries no (or the wrong default) datalayout --
# that's exactly the bug this phase exists to fix.
check_bad_triple() {
    local triple="$1"
    local out err status
    err="$("$TSLANG" --emit=llvm -mm=none --no-default-lib -mtriple="$triple" "$work/dl.ts" \
        -o "$work/bad.ll" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "FAIL $triple: expected a non-zero exit for an unresolvable triple, got 0"
        fail=1
    elif ! printf '%s' "$err" | grep -q -- "$triple"; then
        echo "FAIL $triple: expected the error to name the triple, got: $err"
        fail=1
    else
        echo "ok   $triple (rejected as expected)"
    fi
}

check_bad_triple "not-a-real-triple"

exit "$fail"
