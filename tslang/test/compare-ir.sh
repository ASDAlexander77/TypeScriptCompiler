#!/usr/bin/env bash
# Compiles every test in tslang/test/tester/tests to LLVM IR with the given compiler and triple,
# masks tslang's run-to-run naming noise, and diffs each file against a stored masked baseline.
# Prints one line per file that differs or that newly fails/succeeds, then a summary.
#   compare-ir.sh <tslang.exe> <triple> <baseline-dir> <out-dir>
set -u
TSLANG="${1:?tslang.exe}"; TRIPLE="${2:?triple}"; BASE="${3:?baseline dir}"; OUT="${4:?out dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TESTS="$HERE/tester/tests"
MASK="${MASK:-$HERE/mask-ir.sh}"
if [ ! -x "$MASK" ]; then
    echo "compare-ir.sh: mask '$MASK' is missing or not executable; without it every file would differ" >&2
    exit 2
fi
mkdir -p "$OUT"
same=0; differ=0; newfail=0; newpass=0
for f in "$TESTS"/*.ts; do
    b="$(basename "$f" .ts)"
    if "$TSLANG" --emit=llvm -mm=gc --no-default-lib -mtriple="$TRIPLE" "$f" -o "$OUT/$b.raw.ll" >/dev/null 2>&1; then
        "$MASK" "$OUT/$b.raw.ll" > "$OUT/$b.ll"
        if [ ! -f "$BASE/$b.ll" ]; then echo "NEWPASS $b"; newpass=$((newpass+1))
        elif diff -q "$BASE/$b.ll" "$OUT/$b.ll" >/dev/null; then same=$((same+1))
        else echo "DIFF    $b"; differ=$((differ+1)); fi
    else
        if [ -f "$BASE/$b.ll" ]; then echo "NEWFAIL $b"; newfail=$((newfail+1)); fi
    fi
done
echo "same=$same differ=$differ newfail=$newfail newpass=$newpass"
