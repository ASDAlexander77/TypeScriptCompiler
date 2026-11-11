#!/usr/bin/env bash
# bench.ps1 for Linux and macOS: builds rtbench.ts (rtbench_own.ts for -mm=own) and
# rtbench_static.ts under every memory model and the Rust and C++ ports, checks that they all print
# the same checksum, then times them: a warm-up run each, then RUNS rounds in which every build
# runs once, so that the machine's drift falls on every build alike. Reports min/median/max wall
# time in ms.
#
#   TSLANG=<path to tslang> [RUNS=15] bash bench.sh
#
# tslang finds its libraries through GC_LIB_PATH and TSLANG_LIB_PATH (or beside itself, as a
# release archive has them).
set -u
here=$(cd "$(dirname "$0")" && pwd)
tslang=${TSLANG:-$here/../../../__build/tslang/ninja/release/bin/tslang}
runs=${RUNS:-15}
out=$here/out
mkdir -p "$out"

names=()
cmds=()
add() { names+=("$1"); cmds+=("$2"); }

flags="--emit=exe --opt --opt_level=3 --no-default-lib"
for mm in gc rc none own; do
    src=rtbench.ts
    if [ $mm = own ]; then src=rtbench_own.ts; fi
    if "$tslang" $flags --mm=$mm "$here/$src" -o "$out/rtb_$mm" >"$out/rtb_$mm.log" 2>&1; then
        add "tslang $mm" "$out/rtb_$mm"
    else
        echo "tslang $mm does not compile, see $out/rtb_$mm.log"
    fi
done
for mm in gc rc none own; do
    if "$tslang" $flags --mm=$mm "$here/rtbench_static.ts" -o "$out/rtbs_$mm" >"$out/rtbs_$mm.log" 2>&1; then
        add "tslang $mm, static" "$out/rtbs_$mm"
    else
        echo "tslang $mm, static does not compile, see $out/rtbs_$mm.log"
    fi
done

if command -v cargo >/dev/null; then
    if (cd "$here/rust" && cargo build --release --quiet); then
        add "rust idiomatic" "$here/rust/target/release/idiomatic"
        add "rust static" "$here/rust/target/release/static_dispatch"
        add "rust Rc-boxed" "$here/rust/target/release/boxed"
    fi
fi
if command -v clang++ >/dev/null; then
    for f in idiomatic shared; do
        clang++ -std=c++20 -O3 -o "$out/cpp_$f" "$here/cpp/$f.cpp" && add "c++ $f (clang)" "$out/cpp_$f"
    done
fi

now_us() { echo $(($(date +%s%N) / 1000)); }

checks=()
for i in "${!names[@]}"; do
    # the warm-up run gives the checksum
    if ${cmds[$i]} | grep -q -e '38837369' -e 'low digits: 369'; then checks+=(ok); else checks+=(MISMATCH); fi
    : >"$out/times_$i"
done
for _ in $(seq "$runs"); do
    for i in "${!names[@]}"; do
        t0=$(now_us); ${cmds[$i]} >/dev/null; t1=$(now_us)
        echo $((t1 - t0)) >>"$out/times_$i"
    done
done
for i in "${!names[@]}"; do
    sort -n "$out/times_$i" | awk -v name="${names[$i]}" -v check="${checks[$i]}" '
        { t[NR] = $1 / 1000 }
        END { printf "%-22s min %8.1f ms   median %8.1f ms   max %8.1f ms   checksum %s\n",
                  name, t[1], t[int((NR + 1) / 2)], t[NR], check }'
    rm -f "$out/times_$i"
done
