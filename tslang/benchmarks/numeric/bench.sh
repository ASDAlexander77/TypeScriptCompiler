#!/usr/bin/env bash
# Builds nbench.ts under every memory model and the Rust and C++ ports, checks that they all print
# the same six check values, then times each kernel by itself: a warm-up run and RUNS timed runs,
# reporting the median wall time in ms.
#
#   TSLANG=<path to tslang> [RUNS=9] [JIT_LIB=<libTypeScriptRuntime.so>] bash bench.sh
#
# tslang finds its libraries through GC_LIB_PATH and TSLANG_LIB_PATH (or beside itself, as a
# release archive has them). With JIT_LIB the program is also timed under --emit=jit, the JIT cache
# warm, which adds the compiler's start-up to every figure.
set -u
here=$(cd "$(dirname "$0")" && pwd)
tslang=${TSLANG:-$here/../../../__build/tslang/ninja/release/bin/tslang}
runs=${RUNS:-9}
out=$here/out
mkdir -p "$out"
kernels="saxpy triad stencil dot matmul hash"
expected="saxpy: 346503 triad: 527762 stencil: 218331 dot: 369289 matmul: 615082 hash: 102784"

names=()
cmds=()
add() { names+=("$1"); cmds+=("$2"); }

flags="--opt --opt_level=3 --no-default-lib"
for mm in gc rc none own; do
    if "$tslang" --emit=exe $flags --mm=$mm "$here/nbench.ts" -o "$out/nb_$mm" >"$out/nb_$mm.log" 2>&1; then
        add "tslang $mm" "$out/nb_$mm"
    else
        echo "tslang $mm does not compile, see $out/nb_$mm.log"
    fi
    if [ -n "${JIT_LIB:-}" ]; then
        add "tslang $mm (jit)" "$tslang --emit=jit $flags --mm=$mm --shared-libs=$JIT_LIB --jit-cache-dir=$out/jit_$mm $here/nbench.ts"
    fi
done

if command -v cargo >/dev/null; then
    (cd "$here/rust" && cargo build --release --quiet) && add "rust" "$here/rust/target/release/nbench"
fi
if command -v clang++ >/dev/null; then
    clang++ -std=c++20 -O3 -ffp-contract=off -o "$out/cpp_clang" "$here/cpp/nbench.cpp" && add "c++ (clang)" "$out/cpp_clang"
fi

now_ms() { echo $(($(date +%s%N) / 1000)); }

printf '%-20s' build
for k in $kernels; do printf '%9s' "$k"; done
printf '   %s\n' checksums

for i in "${!names[@]}"; do
    cmd=${cmds[$i]}
    # the whole program once: the check values, and under the JIT the object for the cache
    got=$($cmd | grep -v -e '^start' -e '^done' | tr '\n' ' ' | sed 's/ *$//')
    printf '%-20s' "${names[$i]}"
    for k in $kernels; do
        $cmd "$k" >/dev/null
        times=()
        for _ in $(seq "$runs"); do
            t0=$(now_ms); $cmd "$k" >/dev/null; t1=$(now_ms)
            times+=($((t1 - t0)))
        done
        median=$(printf '%s\n' "${times[@]}" | sort -n | sed -n "$(((runs + 1) / 2))p")
        printf '%9s' "$(awk "BEGIN { printf \"%.1f\", $median / 1000 }")"
    done
    if [ "$got" = "$expected" ]; then echo "   ok"; else echo "   MISMATCH: $got"; fi
done
