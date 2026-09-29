// -mm=own, phase 1: a move inside a loop body whose source is declared in that body. Each
// iteration makes a new `a`, so the move is not "made outside the loop" (spec 2.6).
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        let a = new C(i % 10);
        a.v.push(i);
        let b = a;
        t += b.x + b.v.length;
    }

    assert(t == 550000);
    print("done.");
}
