import './export_gc_single_collector'

// One collector per process. The shared library builds strings that only THIS module's array
// holds, then allocates hard with different content. If the library's collector is not the
// importer's - two static Boehms, or a JIT whose TypeScriptRuntime.dll carries its own - it
// cannot see the array, frees the strings and reuses their memory. The churn must differ from
// what is held, or a freed string is rebuilt with identical bytes and reads back correct
// (reference-counting-evaluation.md 9.76). See docs/single-gc-collector-design.md.

function main() {
    const N = 2000;

    let held: string[] = [];
    for (let i = 0; i < N; i++) {
        held.push(G.makeKey(i));
    }

    const total = G.churn(300000);

    let bad = 0;
    for (let i = 0; i < N; i++) {
        if (held[i] != `key-${i}-end`) bad = bad + 1;
    }

    print("bad:", bad);
    assert(bad == 0, "strings built by the shared library were freed while still held");
    assert(total > 0, "churn");

    print("done.");
}
