#!/usr/bin/env bash
# Corpus probe: compile and run every file in tslang/test/tester/tests as a single-file exe, for
# x64 and i686 under the gc, rc and none memory models, with the suite's flags. Hand-run, like the
# check-*.sh scripts; the result is a TSV to compare against a baseline taken before a change.
#
#   probe-corpus.sh <tslang.exe> <out-dir> [jobs]
#       Writes <out-dir>/results.tsv with the columns: name arch mm stage code
#       stage is pass, compile, link, run or timeout; code is the compile or run exit code.
#
#   probe-corpus.sh --compare <before.tsv> <after.tsv>
#       Prints every row that passed before and does not pass after, then a summary line.
#       Exits non-zero if there is any such row.
#
# About 100 files per model fail at x64 too: they are multi-file tests that need the test runner.
# "No new failures" is judged against a baseline, not against zero.
set -u

if [ "${1:-}" = "--compare" ]; then
    before="${2:?usage: probe-corpus.sh --compare <before.tsv> <after.tsv>}"
    after="${3:?usage: probe-corpus.sh --compare <before.tsv> <after.tsv>}"
    for f in "$before" "$after"; do
        [ -f "$f" ] || { echo "FAIL missing prerequisite: $f"; exit 2; }
    done
    awk -F'\t' '
        NR == FNR { if ($4 == "pass") passed[$1 FS $2 FS $3] = 1; next }
        { key = $1 FS $2 FS $3; seen[key] = 1
          if ((key in passed) && $4 != "pass") { print "REGRESSED\t" $0; bad++ }
          if ($4 == "pass" && !(key in passed)) fixed++ }
        END {
          for (k in passed) if (!(k in seen)) { split(k, p, FS); print "MISSING\t" p[1] "\t" p[2] "\t" p[3]; bad++ }
          printf "summary: %d passed before and not after, %d newly passing\n", bad + 0, fixed + 0
          exit bad > 0 ? 1 : 0
        }' "$before" "$after"
    exit $?
fi

# One case, run by xargs below: probe-corpus.sh --one <out-dir> <tslang.exe> <test.ts> <arch> <mm>
if [ "${1:-}" = "--one" ]; then
    out="$2"; T="$3"; f="$4"; arch="$5"; mm="$6"
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    name="$(basename "$f" .ts)"
    case "$arch" in
        x64) triple=x86_64-pc-windows-msvc
             libs=(--tslang-lib-path="$REPO/__build/tslang/windows-msbuild-2026-release/lib") ;;
        x86) triple=i686-pc-windows-msvc
             libs=(--tslang-lib-path="$REPO/__build/tslang-runtime/release") ;;
    esac
    libs+=(--gc-lib-path="$REPO/3rdParty/gc/x64/release/lib")
    d="$out/cases/$arch/$mm/$name"
    mkdir -p "$d"
    ( cd "$d" && "$T" --emit=exe --opt --opt_level=3 --no-default-lib --entry-point -mm="$mm" \
        -mtriple="$triple" "${libs[@]}" "$f" -o "$d/$name.exe" > "$d/compile.txt" 2>&1 )
    c=$?
    if [ $c -ne 0 ]; then
        if grep -q "LNK" "$d/compile.txt"; then stage=link; else stage=compile; fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$arch" "$mm" "$stage" "$c"
    else
        ( cd "$d" && timeout 30 "$d/$name.exe" > "$d/run.txt" 2>&1 )
        r=$?
        if [ $r -eq 0 ]; then stage=pass; elif [ $r -eq 124 ]; then stage=timeout; else stage=run; fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$arch" "$mm" "$stage" "$r"
    fi
    rm -f "$d"/*.exe "$d"/*.obj "$d"/*.o "$d"/*.pdb "$d"/*.ilk "$d"/*.lib "$d"/*.exp 2>/dev/null
    exit 0
fi

TSLANG="${1:?usage: probe-corpus.sh <tslang.exe> <out-dir> [jobs] | --compare <before.tsv> <after.tsv>}"
OUT="${2:?usage: probe-corpus.sh <tslang.exe> <out-dir> [jobs]}"
JOBS="${3:-12}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TESTS="$REPO/tslang/test/tester/tests"

# These point at a stale install on some machines and would override the explicit lib flags.
unset GC_LIB_PATH TSLANG_LIB_PATH GC_SHARED_LIB_PATH DEFAULT_LIB_PATH

missing=0
need() { [ -e "$1" ] || { echo "FAIL missing prerequisite: $1"; missing=1; }; }
need "$TSLANG"
need "$TESTS"
need "$REPO/3rdParty/gc/x64/release/lib/gc.lib"
need "$REPO/3rdParty/gc/x64/release/lib/x86"
need "$REPO/__build/tslang/windows-msbuild-2026-release/lib/TypeScriptAsyncRuntime.lib"
need "$REPO/__build/tslang-runtime/release/x86"
command -v timeout >/dev/null 2>&1 || { echo "FAIL missing prerequisite: timeout"; missing=1; }
command -v xargs >/dev/null 2>&1 || { echo "FAIL missing prerequisite: xargs"; missing=1; }
[ $missing -eq 0 ] || exit 2

TSLANG="$(cd "$(dirname "$TSLANG")" && pwd)/$(basename "$TSLANG")"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

for f in "$TESTS"/*.ts; do
    for arch in x64 x86; do
        for mm in gc rc none; do
            printf '%s\0%s\0%s\0' "$f" "$arch" "$mm"
        done
    done
done | xargs -0 -n 3 -P "$JOBS" bash "$SELF" --one "$OUT" "$TSLANG" > "$OUT/results.unsorted"

sort "$OUT/results.unsorted" > "$OUT/results.tsv"
rm -f "$OUT/results.unsorted"

echo "arch mm pass total"
awk -F'\t' '{ t[$2 " " $3]++; if ($4 == "pass") p[$2 " " $3]++ }
    END { for (k in t) printf "%s %d %d\n", k, p[k] + 0, t[k] }' "$OUT/results.tsv" | sort
echo "wrote $OUT/results.tsv"
