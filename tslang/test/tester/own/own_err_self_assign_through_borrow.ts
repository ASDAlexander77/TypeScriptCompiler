// A slot given back a value that borrows its own earlier one, after a move of it on one path only:
// -mm=own refuses it, and says so - the inference walked the slot's reads round that store for
// ever, so the compiler never finished (fuzz 946038).
class N { x: number = 0; }
function mk(i: number): N { const n = new N(); n.x = i; return n; }
function g11(p12: N): N { return p12; }
function main() {
    let a13: N[] = [mk(4)];
    let n14: N = mk(9);
    if (n14.x > 1) {
        a13.push(n14);
    }
    n14 = g11((n14 = mk(1), n14));
}
