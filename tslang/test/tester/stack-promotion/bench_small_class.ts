// Benchmark (not registered with ctest): ten million short-lived two-field objects.
class Pair {
    a: number;
    b: number;
    constructor(a: number, b: number) {
        this.a = a;
        this.b = b;
    }
}

function run(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const p = new Pair(i, 1);
        t += p.a + p.b;
    }
    return t;
}

function main() {
    const t = run(10000000);
    assert(t == 50000005000000);
    print("done.");
}
