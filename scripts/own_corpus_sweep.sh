#!/usr/bin/env bash
# The -mm=own corpus sweep (own rc fallback design, "Measuring"): every file of
# tslang/test/tester/tests compiled under gc, under own, and under own with --own-fallback=count, each
# at the --emit given (mlir-affine measures inference alone; llvm the whole pipeline). Prints the
# files each accepts, the files the flag newly accepts, the warnings per file, and each file own
# still refuses with the flag, by its first error.
#
#   bash scripts/own_corpus_sweep.sh [emit] [tslang]
#     emit    mlir-affine (default) or llvm
#     tslang  the compiler (default: the worktree's Windows Release build)
#
# Every compile writes into a scratch folder (-o): --emit=llvm without -o writes <source>.ll beside
# the test and overwrites the tracked .ll files of tslang/test/tester/tests.
set -u

emit="${1:-mlir-affine}"
root="$(cd "$(dirname "$0")/.." && pwd)"
tslang="${2:-$root/__build/tslang/windows-msbuild-2026-release/bin/tslang.exe}"
tests="$root/tslang/test/tester/tests"
jobs="${SWEEP_JOBS:-8}"

if [[ ! -x "$tslang" ]]; then
    echo "no compiler at $tslang" >&2
    exit 1
fi

scratch="$(mktemp -d "${TMPDIR:-/tmp}/own_sweep.XXXXXX")"
echo "emit=$emit compiler=$tslang scratch=$scratch"

opt=()
if [[ "$emit" == "llvm" ]]; then
    opt=(--opt --opt_level=3)
fi

# one compile: <mode> <file>; writes <scratch>/<mode>/<name>.{out,log,rc}
compile_one() {
    local mode="$1" file="$2"
    local name dir
    name="$(basename "$file" .ts)"
    dir="$scratch/$mode"
    local flags=()
    case "$mode" in
        gc) flags=(-mm=gc) ;;
        own) flags=(-mm=own) ;;
        own-count) flags=(-mm=own --own-fallback=count) ;;
    esac

    "$tslang" --emit="$emit" --no-default-lib "${opt[@]}" "${flags[@]}" -o "$dir/$name.out" "$file" \
        > "$dir/$name.log" 2>&1
    echo $? > "$dir/$name.rc"
    # the printed IR is not kept: only the diagnostics
    grep -E "(error|warning): " "$dir/$name.log" > "$dir/$name.diag" 2>/dev/null
    rm -f "$dir/$name.log" "$dir/$name.out"
}
export -f compile_one
export scratch tslang emit
export opt_flags="${opt[*]:-}"

for mode in gc own own-count; do
    mkdir -p "$scratch/$mode"
    find "$tests" -maxdepth 1 -name "*.ts" -print0 |
        xargs -0 -P "$jobs" -I{} bash -c 'opt=($opt_flags); compile_one '"$mode"' "$1"' _ {}
done

accepted() {
    local mode="$1"
    for rc in "$scratch/$mode"/*.rc; do
        [[ "$(cat "$rc")" == "0" ]] && basename "$rc" .rc
    done | sort
}

accepted gc > "$scratch/gc.accepted"
accepted own > "$scratch/own.accepted"
accepted own-count > "$scratch/own-count.accepted"

total=$(find "$tests" -maxdepth 1 -name "*.ts" | wc -l)
echo
echo "files:                  $total"
echo "gc accepts:             $(wc -l < "$scratch/gc.accepted")"
echo "own accepts:            $(wc -l < "$scratch/own.accepted")"
echo "own + count accepts:    $(wc -l < "$scratch/own-count.accepted")"

echo
echo "newly accepted with --own-fallback=count (warnings):"
comm -13 "$scratch/own.accepted" "$scratch/own-count.accepted" | while read -r name; do
    echo "  $name ($(grep -c "counts this object here" "$scratch/own-count/$name.diag"))"
done

echo
echo "accepted without the flag, refused with it (must be none):"
comm -23 "$scratch/own.accepted" "$scratch/own-count.accepted" | sed 's/^/  /'

echo
echo "still refused with --own-fallback=count (gc accepts), by first error:"
comm -12 "$scratch/gc.accepted" <(ls "$scratch/own-count"/*.rc | xargs -n1 basename | sed 's/\.rc$//' | sort) |
    comm -23 - "$scratch/own-count.accepted" | while read -r name; do
    first="$(grep -m1 "error: " "$scratch/own-count/$name.diag" | sed -E 's/^.*error: //')"
    printf '%s\t%s\n' "$name" "${first:-<no error line>}"
done > "$scratch/refused.tsv"
sed 's/^/  /' "$scratch/refused.tsv"

echo
echo "first errors, names elided, by count:"
cut -f2 "$scratch/refused.tsv" | sed -E "s/'[^']*'/'X'/g" | sort | uniq -c | sort -rn | sed 's/^/  /'

echo
echo "results kept in $scratch"
