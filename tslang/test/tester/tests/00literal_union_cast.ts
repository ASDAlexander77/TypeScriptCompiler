// regression test: a union of literals, as a ternary of two literals is, converts to another
// numeric type through its literals' type.
//
// bug found and fixed: `c ? 1 : 0` is the union `0 | 1`; cast straight to `index` (a variable, a
// parameter, a tuple field) it reached the lowering as a union, and `llvm.sext` of the union failed
// with no source location.

type Vec = [items: [_0: Opaque | null], count: index];

function make(item: Opaque | null): Vec {
    return [[item], item ? 1 : 0];
}

function takeIndex(n: index): index {
    return n;
}

function main() {
    let c = true;
    let off = false;

    let x: index = c ? 1 : 0;
    assert(x == 1);

    let y: index = off ? 1 : 0;
    assert(y == 0);

    assert(takeIndex(c ? 7 : 3) == 7);

    let n: number = c ? 1 : 0;
    assert(n == 1);

    let w: s64 = off ? 2 : 5;
    assert(w == 5);

    assert(make(null).count == 0);

    print("done.");
}
