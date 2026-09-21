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
#
# Phase 4b gate: async/await programs, including `for await`, at i686. The async runtime needs
# Boehm's thread API, so these link only under -mm=gc (rc and none fail to link at x64 too; out of
# scope). Each corpus file asserts internally, so exiting 0 is the check, as in check-x86-eh.sh's
# exception corpus loop. await_order.ts is the minimal repro from the phase 4b plan and gets an
# exact-output check because its whole point is the print ORDER around the await.
set -u
TSLANG="${1:?usage: check-x86-run.sh <path to tslang.exe>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PROGRAMS="$REPO/tslang/test/x86"
TESTS="$REPO/tslang/test/tester/tests"
READOBJ="$REPO/3rdParty/llvm/x64/release/bin/llvm-readobj.exe"

# Hermetic: the compiler falls back to these when a flag is missing, and they may name stale builds.
unset GC_LIB_PATH GC_SHARED_LIB_PATH TSLANG_LIB_PATH DEFAULT_LIB_PATH LLVM_LIB_PATH

# The documented layout: build_gc_release_vs_x86.bat copies the x86 gc.lib into the x86
# subdirectory of the x64 lib directory, so the flag x64 programs use serves x86 as well.
GC_DIR="$REPO/3rdParty/gc/x64/release/lib"
GC_X86="$GC_DIR/x86/gc.lib"
RT_DIR="$REPO/__build/tslang-runtime/release"
RT_X86="$RT_DIR/x86/TypeScriptAsyncRuntime.lib"

# The async corpus cases (gc only), from test/tester/tests. They are built with --entry-point to
# mirror the suite; for --emit=exe it changes nothing, it matters only to the --emit=llvm IR checks.
ASYNC_CORPUS=(00async_await 00async_gc_threading 00async_result_types 00owned_async
              00for_await 00for_await_yield)

for f in "$TSLANG" "$READOBJ" "$GC_X86" "$RT_X86" "$PROGRAMS/hello.ts" "$PROGRAMS/gc_stress.ts" \
         "$PROGRAMS/await_order.ts"; do
    if [ ! -f "$f" ]; then
        echo "FAIL missing prerequisite: $f"
        exit 1
    fi
done
for name in "${ASYNC_CORPUS[@]}"; do
    if [ ! -f "$TESTS/$name.ts" ]; then
        echo "FAIL missing prerequisite: $TESTS/$name.ts"
        exit 1
    fi
done
# Without timeout a hang looks identical to a run in progress rather than a FAIL.
command -v timeout >/dev/null 2>&1 || { echo "FAIL missing prerequisite: timeout"; exit 1; }

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
        gc_stress)   printf '499500\n3090\n' ;;
        # Confirmed at x64 (docs/superpowers/plans/2026-09-22-32-bit-phase-4b-async.md): a
        # non-async main resumes past `await g()` before g()'s body runs on the async runtime.
        await_order) printf 'start\nafter await\nin g\n' ;;
    esac
}

fail=0
# run_case <program> <model> <extra tslang args>...
run_case() {
    local prog="$1" mm="$2"
    shift 2
    local name="$prog -mm=$mm" exe="$work/$prog-$mm.exe" err out status machine
    err="$("$TSLANG" --emit=exe --opt -mm="$mm" --no-default-lib --entry-point -mtriple=i686-pc-windows-msvc \
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
    # The Windows CRT writes stdout in text mode, so lines end in CR LF. A 30s timeout catches a
    # hang (distinct from the crashes these cases are otherwise looking for) as a FAIL, not a wait.
    out="$(timeout 30 "$exe" 2>"$work/stderr.txt" | tr -d '\r'; echo "exit=${PIPESTATUS[0]}")"
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

# --- Phase 4c: installer detection --------------------------------------------------------
# Windows asks for elevation before starting a 32-bit exe that has no requestedExecutionLevel
# manifest and whose name contains "setup", "install", "update" or "patch"; unelevated, it never
# starts. tslang embeds an asInvoker manifest in every Windows x86 exe, so this name must run.
# (The suite hit it with export-import-class-abstract-virtual-dispatch: "dispatch" holds "patch".)
installer_name="my_setup_patch"
installer_exe="$work/$installer_name.exe"
err="$("$TSLANG" --emit=exe --opt -mm=none --no-default-lib --entry-point -mtriple=i686-pc-windows-msvc \
    "$RT_FLAG" "$PROGRAMS/hello.ts" -o "$installer_exe" 2>&1 >/dev/null)"
status=$?
if [ "$status" -ne 0 ] || [ ! -f "$installer_exe" ]; then
    echo "FAIL $installer_name.exe: compile failed (exit $status): $err"
    fail=1
else
    # Without the manifest the run fails at once ("Permission denied", exit 126): CreateProcess
    # refuses with ERROR_ELEVATION_REQUIRED rather than prompting.
    out="$(timeout 30 "$installer_exe" 2>&1 | tr -d '\r'; echo "exit=${PIPESTATUS[0]}")"
    if [ "$out" != "$(expected hello; echo "exit=0")" ]; then
        echo "FAIL $installer_name.exe: runs unelevated and prints the hello text"
        printf '%s\n' "$out" | sed 's/^/    got: /'
        fail=1
    elif ! "$READOBJ" --coff-resources "$installer_exe" | grep -q 'Type: MANIFEST (ID 24)'; then
        echo "FAIL $installer_name.exe: runs, but has no RT_MANIFEST resource"
        fail=1
    else
        echo "ok   $installer_name.exe (x86, asInvoker manifest) runs unelevated"
    fi
fi

# await_order.ts links only under -mm=gc: async needs Boehm's thread API, and rc/none fail to
# link even at x64 (out of scope; see the plan's Global Constraints).
run_case await_order gc "$GC_FLAG" "$RT_FLAG"

# --- Phase 4b: async corpus files run at i686 under gc -------------------------------------
# run_async_corpus <name> -> builds test/tester/tests/<name>.ts at i686, -mm=gc, --entry-point;
# checks the I386 machine and that a 30s-timeout run exits 0. The file's own assert()s are the
# correctness check; only the exit code is verified here, as check-x86-eh.sh does for its
# exception corpus.
run_async_corpus() {
    local name="$1"
    local label="$name (async, -mm=gc)" exe="$work/async-$name.exe" err status machine run_status
    err="$("$TSLANG" --emit=exe --opt -mm=gc --no-default-lib --entry-point -mtriple=i686-pc-windows-msvc \
        "$GC_FLAG" "$RT_FLAG" "$TESTS/$name.ts" -o "$exe" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -ne 0 ] || [ ! -f "$exe" ]; then
        echo "FAIL $label: compile failed (exit $status): $err"
        fail=1
        return
    fi
    machine="$("$READOBJ" --file-headers "$exe" | grep -m1 'Machine:')"
    if ! printf '%s' "$machine" | grep -q 'IMAGE_FILE_MACHINE_I386 (0x14C)'; then
        echo "FAIL $label: expected an I386 (0x14C) executable, got: $machine"
        fail=1
        return
    fi
    timeout 30 "$exe" >"$work/stdout.txt" 2>"$work/stderr.txt"
    run_status=$?
    if [ "$run_status" -ne 0 ]; then
        echo "FAIL $label: expected exit 0, got $run_status"
        sed 's/^/    stdout: /' "$work/stdout.txt"
        sed 's/^/    stderr: /' "$work/stderr.txt"
        fail=1
        return
    fi
    echo "ok   $label"
}

for name in "${ASYNC_CORPUS[@]}"; do
    run_async_corpus "$name"
done

# --- Phase 4b: frame allocator width (i686) -------------------------------------------------
# Upstream ConvertAsyncToLLVM hardcodes the coroutine frame allocator's declaration to
# aligned_alloc(i64, i64), whatever the target. aligned_alloc takes size_t, which is pointer
# width, so at i686 the callee reads (alignment, size=0) and the frame overflows its block.
# The repair pass (task 2/3) must retype the declaration to the pointer width: GC_memalign is
# GCPass's rename of aligned_alloc when a collector is linked in.
emit_await_order_ir() {
    local mm="$1"
    local out="$work/await_order.$mm.ll" err status
    err="$("$TSLANG" --emit=llvm --opt -mm="$mm" --no-default-lib --entry-point -mtriple=i686-pc-windows-msvc \
        "$PROGRAMS/await_order.ts" -o "$out" 2>&1 >/dev/null)"
    status=$?
    if [ "$status" -ne 0 ] || [ ! -f "$out" ]; then
        echo "FAIL x86 IR -mm=$mm: await_order compiles (exit $status): $err"
        fail=1
        return 1
    fi
    printf '%s' "$out"
}

# Only the parameter TYPES are checked: once aligned_alloc has the target's size_t signature,
# LLVM recognizes it as the C library function and adds attributes to the declaration, e.g.
# `declare noalias noundef ptr @aligned_alloc(i32 allocalign noundef, i32 noundef)`.
declared_with_i32_params() {
    grep -Eq "^declare .*ptr @$1\(i32( [^,]*)?, i32( [^)]*)?\)" "$2"
}

if out="$(emit_await_order_ir gc)"; then
    if declared_with_i32_params GC_memalign "$out"; then
        echo "ok   x86 IR -mm=gc: frame allocator is GC_memalign(i32, i32)"
    else
        echo "FAIL x86 IR -mm=gc: frame allocator is GC_memalign(i32, i32)"
        grep -m1 -E '@GC_memalign\(|@aligned_alloc\(' "$out" | sed 's/^/    got: /'
        fail=1
    fi
fi

if out="$(emit_await_order_ir none)"; then
    if declared_with_i32_params aligned_alloc "$out"; then
        echo "ok   x86 IR -mm=none: frame allocator is aligned_alloc(i32, i32)"
    else
        echo "FAIL x86 IR -mm=none: frame allocator is aligned_alloc(i32, i32)"
        grep -m1 -E '@aligned_alloc\(' "$out" | sed 's/^/    got: /'
        fail=1
    fi
fi

exit "$fail"
