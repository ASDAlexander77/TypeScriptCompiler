// A module-level global never given another value is immortal under -mm=own, as a class static
// is (own_immortal_global): its initializer is its own region, which the lowering makes a
// constructor of, and that constructor's store marks the block. `e` is given `c`'s value.
class V {
    constructor(public x: number) {}
}

const c = new V(1);
let e = c;

function f() { return c; }   // returned
function h() { return e; }

function g(n: number) { return n > 1 ? c : new V(n); }   // immortal or fresh

function main() {
    let s: number = 0;
    for (let i = 0; i < 100000; i++) { s += f().x + h().x + g(i % 3).x; }
    print(s, c.x, e.x);   // 266666 1 1
    assert(s == 266666 && c.x == 1 && e.x == 1);
    print("done.");
}
