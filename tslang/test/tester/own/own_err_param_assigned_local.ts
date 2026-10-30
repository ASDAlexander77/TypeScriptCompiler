// -mm=own rejects: `p = n` puts `n`'s object in a parameter's slot, which owns nothing, so the
// object is neither taken from `n` nor counted again - and `n = mk(9)` frees it while `p` still
// reads it. Own accepted this (it reported an assigned parameter only where a retain stood for the
// store, and a store into a parameter has none), and `p.x` read freed memory (mm_fuzz seed 70377).
class N {
    constructor(public x: number) {}
}

function mk(i: number): N {
    return new N(i);
}

function f(p: N): number {
    let n: N = mk(2);
    p = n;
    n = mk(9);
    return p.x;
}

function main() {
    print(f(mk(5)));
}
