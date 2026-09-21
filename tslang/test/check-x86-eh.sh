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

exit "$fail"
