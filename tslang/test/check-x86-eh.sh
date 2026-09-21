#!/usr/bin/env bash
# Checks for 32-bit Windows (i686-pc-windows-msvc) exception handling. Hand-run, like the other
# check-x86-*.sh scripts: tslang/test/check-x86-eh.sh <path/to/tslang.exe>
set -u
TSLANG="${1:?usage: check-x86-eh.sh <tslang.exe>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TESTS="$REPO/tslang/test/tester/tests"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unset GC_LIB_PATH TSLANG_LIB_PATH GC_SHARED_LIB_PATH DEFAULT_LIB_PATH
fail=0
ok()   { echo "ok   $1"; }
bad()  { echo "FAIL $1"; fail=1; }
need() { [ -e "$1" ] || { echo "FAIL missing prerequisite: $1"; exit 1; }; }
need "$TSLANG"
# Without timeout every run would report exit 127, which looks like a real failure mode.
command -v timeout >/dev/null || { echo "FAIL missing prerequisite: timeout"; exit 1; }

# emit_ir <triple> <test-name> -> path of the .ll
emit_ir() {
  local out="$TMP/$2.$1.ll"
  "$TSLANG" --emit=llvm --opt -mm=none --no-default-lib -mtriple="$1" "$TESTS/$2.ts" -o "$out" 2>"$out.err" \
    || { echo "compile failed: $1 $2" >&2; cat "$out.err" >&2; return 1; }
  echo "$out"
}

# --- Task 1: EH cross-references ---------------------------------------------------------
# The i32 cross-references inside _TI (ThrowInfo), _CT (CatchableType) and _CTA
# (CatchableTypeArray) are global initializers, so they print as constant expressions:
#   x64: i32 trunc (i64 sub (i64 ptrtoint (ptr @X to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)
#   x86: i32 ptrtoint (ptr @X to i32)
x86ll="$(emit_ir i686-pc-windows-msvc 00try_catch)" || bad "x86 IR: 00try_catch compiles"
x64ll="$(emit_ir x86_64-pc-windows-msvc 00try_catch)" || bad "x64 IR: 00try_catch compiles"
# Names may be quoted (@"_CT??_R0H@84"), so each prefix allows an optional quote.
eh_globals=("_CT[?][?]" "_CTA" "_TI")
label() { case "$1" in _CTA) echo "CatchableTypeArray";; _TI) echo "ThrowInfo";; *) echo "CatchableType";; esac; }
if [ -n "${x86ll:-}" ]; then
  grep -q "__ImageBase" "$x86ll" && bad "x86 IR: no __ImageBase" || ok "x86 IR: no __ImageBase"
  for g in "${eh_globals[@]}"; do
    grep -Eq "^@\"?$g[^ ]* = .* i32 ptrtoint \(ptr @[^ ]+ to i32\)" "$x86ll" \
      && ok "x86 IR: $(label "$g") holds absolute i32 references" || bad "x86 IR: $(label "$g") holds absolute i32 references"
  done
fi
if [ -n "${x64ll:-}" ]; then
  for g in "${eh_globals[@]}"; do
    grep -Eq "^@\"?$g[^ ]* = .* i32 trunc \(i64 sub \(i64 ptrtoint \(ptr @[^ ]+ to i64\), i64 ptrtoint \(ptr @__ImageBase to i64\)\) to i32\)" "$x64ll" \
      && ok "x64 IR: $(label "$g") still image-base relative" || bad "x64 IR: $(label "$g") still image-base relative"
  done
fi

# In the x86 object the references must be absolute (IMAGE_REL_I386_DIR32), not image-relative
# (IMAGE_REL_I386_DIR32NB, which is what the RVA arithmetic lowered to).
READOBJ="$REPO/3rdParty/llvm/x64/release/bin/llvm-readobj.exe"
need "$READOBJ"
x86obj="$TMP/00try_catch.i686.obj"
if "$TSLANG" --emit=obj --opt -mm=none --no-default-lib -mtriple=i686-pc-windows-msvc "$TESTS/00try_catch.ts" \
     -o "$x86obj" 2>"$x86obj.err"; then
  "$READOBJ" --relocations "$x86obj" > "$x86obj.rel"
  grep -q "DIR32NB" "$x86obj.rel" && bad "x86 obj: no image-relative (DIR32NB) relocations" \
                                  || ok "x86 obj: no image-relative (DIR32NB) relocations"
  grep -Eq "IMAGE_REL_I386_DIR32 __CTA1H " "$x86obj.rel" && ok "x86 obj: absolute (DIR32) EH relocations" \
                                                         || bad "x86 obj: absolute (DIR32) EH relocations"
else
  bad "x86 obj: 00try_catch compiles"; cat "$x86obj.err"
fi

# --- Task 2: _CxxThrowException is __stdcall on x86 ---------------------------------------
# The decorated symbol is what the linker sees.
for triple in i686-pc-windows-msvc x86_64-pc-windows-msvc; do
  obj="$TMP/throw.$triple.obj"
  if "$TSLANG" --emit=obj --opt -mm=none --no-default-lib -mtriple="$triple" "$TESTS/00try_catch.ts" -o "$obj" 2>"$obj.err"; then
    syms="$("$READOBJ" --symbols "$obj" | grep -o 'Name: .*CxxThrowException.*' | sort -u)"
    case "$triple" in
      i686*)   [ "$syms" = "Name: __CxxThrowException@8" ] && ok "x86 obj: __CxxThrowException@8" \
                                                            || bad "x86 obj: __CxxThrowException@8 (got: $syms)";;
      x86_64*) [ "$syms" = "Name: _CxxThrowException" ] && ok "x64 obj: _CxxThrowException unchanged" \
                                                         || bad "x64 obj: _CxxThrowException unchanged (got: $syms)";;
    esac
  else
    bad "$triple obj: 00try_catch compiles"; cat "$obj.err"
  fi
done

# Every call site must carry the convention too: one left at the C convention is UB, and
# InstCombine turns it into `unreachable`. 00try_catch throws from main (a call), 00try_catch_rethrow
# throws inside a try (an invoke), and 00try_finally gets the catch-all rethrow that
# Win32ExceptionPass synthesizes for a finally. Checked with and without --opt.
for name in 00try_catch 00try_catch_rethrow 00try_finally; do
  for opt in --opt --opt=false; do
    ll="$TMP/cc.$name$opt.ll"
    if "$TSLANG" --emit=llvm $opt -mm=none --no-default-lib -mtriple=i686-pc-windows-msvc "$TESTS/$name.ts" -o "$ll" 2>"$ll.err"; then
      decl="$(grep -c '^declare x86_stdcallcc void @_CxxThrowException(' "$ll")"
      sites="$(grep -Ec '(call|invoke) (x86_stdcallcc )?void @_CxxThrowException\(' "$ll")"
      cdecl="$(grep -E '(call|invoke) void @_CxxThrowException\(' "$ll")"
      if [ "$decl" = 1 ] && [ "$sites" -gt 0 ] && [ -z "$cdecl" ]; then
        ok "x86 IR $opt: $name declares and calls _CxxThrowException as x86_stdcallcc ($sites sites)"
      else
        bad "x86 IR $opt: $name declares and calls _CxxThrowException as x86_stdcallcc (decl=$decl sites=$sites)"
        [ -n "$cdecl" ] && echo "$cdecl"
      fi
    else
      bad "x86 IR $opt: $name compiles"; cat "$ll.err"
    fi
  done
done

# --- Task 3: 32-bit programs throw and catch correctly ------------------------------------
# Each program is linked as a 32-bit exe under every memory model and run with a 20 s limit, so a
# hang is a FAIL. The x86 libraries sit in the x86 subdirectory of the directories x64 uses.
GC_DIR="$REPO/3rdParty/gc/x64/release/lib"
RT_DIR="$REPO/__build/tslang-runtime/release"
need "$GC_DIR/x86/gc.lib"
need "$RT_DIR/x86/TypeScriptAsyncRuntime.lib"
EH_ORDER="$REPO/tslang/test/x86/eh_order.ts"
need "$EH_ORDER"

# run_x86 <label> <model> <file.ts> -> sets $run_out (stdout without CR, then "exit=<code>");
# returns 1 after reporting a FAIL if the build fails or the exe is not I386.
run_x86() {
  local label="$1" mm="$2" src="$3" exe="$TMP/run.$(basename "$3" .ts).$2.exe" err machine
  local libs=("--tslang-lib-path=$RT_DIR")
  [ "$mm" = gc ] && libs+=("--gc-lib-path=$GC_DIR")
  if ! err="$("$TSLANG" --emit=exe --opt -mm="$mm" --no-default-lib -mtriple=i686-pc-windows-msvc \
                "${libs[@]}" "$src" -o "$exe" 2>&1 >/dev/null)" || [ ! -f "$exe" ]; then
    bad "$label: compiles and links"; echo "$err"; return 1
  fi
  machine="$("$READOBJ" --file-headers "$exe" | grep -m1 'Machine:')"
  if ! printf '%s' "$machine" | grep -q 'IMAGE_FILE_MACHINE_I386 (0x14C)'; then
    bad "$label: I386 (0x14C) executable (got: $machine)"; return 1
  fi
  # The Windows CRT writes stdout in text mode, so lines end in CR LF.
  run_out="$(timeout 20 "$exe" 2>"$exe.stderr" | tr -d '\r'; echo "exit=${PIPESTATUS[0]}")"
}

# Unwinding order: finally blocks run innermost-first, a throw from a catch reaches the outer
# try, a rethrow is caught, and execution continues after each.
eh_expected="throw
finally 0
finally 1
finally 2
caught
inner catch
outer catch
rethrow
caught rethrown
done
exit=0"
for mm in gc rc none; do
  run_x86 "x86 run -mm=$mm: eh_order" "$mm" "$EH_ORDER" || continue
  if [ "$run_out" = "$eh_expected" ]; then
    ok "x86 run -mm=$mm: eh_order prints the unwinding trace"
  else
    bad "x86 run -mm=$mm: eh_order prints the unwinding trace"
    printf '%s\n' "$run_out" | sed 's/^/    got: /'
  fi
done

# The exception corpus files assert internally, so exiting 0 is the check.
eh_corpus=(00try_catch 00nested_catch 00try_finally 00try_finally_return 00try_finally_break_continue
           00try_catch_rethrow 00try_catch_mismatch_rethrow 00throw_in_catch 00try_catch_return
           00alloc_in_catch 51exceptions)
for mm in gc rc none; do
  for name in "${eh_corpus[@]}"; do
    run_x86 "x86 run -mm=$mm: $name" "$mm" "$TESTS/$name.ts" || continue
    status="${run_out##*exit=}"
    if [ "$status" = 0 ]; then
      ok "x86 run -mm=$mm: $name exits 0"
    else
      bad "x86 run -mm=$mm: $name exits 0 (exit $status)"
      printf '%s\n' "$run_out" | tail -5 | sed 's/^/    got: /'
    fi
  done
done

exit "$fail"
