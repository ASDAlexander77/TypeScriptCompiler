// Recursion through a function value: f calls a closure that calls f through `ff`. A call
// through a value may reach any function whose address is taken, so f is recursive.
class P {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

let ff = (n: number): number => 0;

function call(g: () => number) {
    return g();
}

function f(n: number): number {
    const p = new P(n);
    if (n > 0) {
        return p.x + call((): number => ff(n - 1));
    }

    return p.x;
}

function main() {
    ff = f;
    assert(f(10) == 55);
    print("done.");
}
