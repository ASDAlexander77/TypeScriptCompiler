// -mm=own, phase 4 rejects: `keep` keeps its parameter, but `main` passes a global, which it does
// not own and cannot move. `reset` gives the global another value, so it is not immortal (spec 25).
// `keep` pushes, which frees nothing: a keeper that may drop (`h.c = c` frees the old `h.c`) is
// rejected at the call already, as a written global lent to it (#639).
class C {
    constructor(public x: number) {}
}

let g = new C(1);

function keep(a: C[], c: C) {
    a.push(c);
}

function reset() {
    g = new C(2);
}

function main() {
    const a: C[] = [];
    keep(a, g);
    print(a[0].x);
    reset();
}
