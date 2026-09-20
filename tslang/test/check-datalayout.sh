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

exit "$fail"
