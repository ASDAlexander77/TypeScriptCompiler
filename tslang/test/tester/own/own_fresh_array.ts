// -mm=own, phase 0: a fresh array per iteration, owned by a local and lent to a call.
function sum(a: number[]) {
    let t: number = 0;
    for (const v of a) t += v;
    return t;
}
function main() {
    let total: number = 0;
    for (let i = 0; i < 1000000; i++) {
        const a: number[] = [i, i + 1, i + 2];
        total += sum(a);
    }
    assert(total == 1500001500000);
    print("done.");
}
