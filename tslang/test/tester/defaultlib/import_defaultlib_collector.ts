import './export_defaultlib_collector'

// Strings built by the library through the default-library DLL, held only by this program's
// array, then churned by the library with different content. See export_defaultlib_collector.ts.

function main() {
    const N = 2000;

    let held: string[] = [];
    for (let i = 0; i < N; i++) {
        held.push(D.makeKey(i));
    }

    const total = D.churn(300000);

    let bad = 0;
    for (let i = 0; i < N; i++) {
        if (held[i] != `${i}`.padStart(12, "k")) bad = bad + 1;
    }

    print("bad:", bad);
    assert(bad == 0, "strings built by the default-library DLL were freed while still held");
    assert(total > 0, "churn");

    print("done.");
}
