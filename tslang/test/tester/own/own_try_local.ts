// -mm=own, phase 0: a throw out of a function with no try leaves its local unreleased - a leak
// rc shares - and must never become a double free.
function f(i: number) {
    const a: number[] = [i, i];
    if (i % 2 == 0) throw i;
    return a.length;
}
function main() {
    let caught = 0;
    for (let i = 0; i < 100000; i++) {
        try { f(i); } catch (e) { caught++; }
    }
    assert(caught == 50000);
    print("done.");
}
