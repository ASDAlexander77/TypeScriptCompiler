// -mm=own, phase 1 rejects: `const b = a` is another read of `a`'s slot, not a declaration of its
// own, so `b` is `a` read later. `a` then moves into `c`, and reading `b` after that reads the
// moved value - here, after `c` has destroyed it. Holding `b` across the move is a borrow (phase 2).
class C {
    x: number;
    constructor(x: number) { this.x = x; }
}
function main() {
    let a = new C(5);
    const b = a;
    let c = a;
    c = new C(0);
    print(b.x, c.x);
    print("done.");
}
