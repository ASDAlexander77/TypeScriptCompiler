// An implementation with fewer parameters than an overload signature of it (TypeScript allows it: the further
// arguments are ignored) takes a call through that overload: "Expected 1 arguments, but got 2".
function f(a: number): number;
function f(a: number, b: boolean): number;
function f(a: number) {
    return a;
}

class W {
    constructor(a: number);
    constructor(a: number, b: string);
    constructor(public a: number) {}

    p(x: number): number;
    p(x: number, y: boolean, z: string): number;
    p(x: number) {
        return x * 2;
    }
}

function main() {
    assert(f(1) == 1 && f(2, true) == 2);
    const w = new W(3, "x");
    assert(w.a == 3 && new W(4).a == 4);
    assert(w.p(5) == 10 && w.p(6, false, "s") == 12);
    print("done.");
}
