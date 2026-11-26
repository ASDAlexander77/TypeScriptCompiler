// One collector per process, with the default library in it. The strings this program holds are
// built by the default library (padStart), and the churn is too (repeat), so under the JIT they
// are allocated by TypeScriptDefaultLib.dll while the array holding them belongs to the JIT'd
// code. If that DLL carries a collector of its own it cannot see the array, and frees them.
// The churn differs from what is held, and the expectation is built with the same method.
// See tslang/docs/single-gc-collector-design.md.

function makeKey(i: number): string {
    return `${i}`.padStart(12, "k");
}

// (also, with the default library in: a function of the program named as a C function the library declares - `log`,
// its `declare function log(x: number): number` - is the program's; it was "redefinition of symbol named 'log'",
// TSLANG 203)
function log(n: number) {
    return n + 1;
}

function main() {
    const N = 2000;

    let held: string[] = [];
    for (let i = 0; i < N; i++) {
        held.push(makeKey(i));
    }

    let total = 0;
    for (let j = 0; j < 300000; j++) {
        let s = "z".repeat(40 + (j % 7));
        total = total + s.length;
    }

    let bad = 0;
    for (let i = 0; i < N; i++) {
        if (held[i] != makeKey(i)) bad = bad + 1;
    }

    print("bad:", bad);
    assert(bad == 0, "strings built by the default library were freed while still held");
    assert(total > 0, "churn");

    assert(log(1) == 2, "the program's log");

    print("done.");
}
