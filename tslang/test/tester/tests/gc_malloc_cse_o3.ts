// Regression test for: at -O3, two GC_malloc(0) calls with identical
// arguments (allocating two distinct empty array fields in the same
// constructor) were incorrectly CSE'd into a single shared allocation,
// aliasing the two fields. Growing one field's backing array then
// corrupted/orphaned the other field's pointer.
//
// Compile at --opt --opt_level=3 to exercise the bug; passes trivially
// at --opt_level=0.

class TwoArrays {
    a: string[];
    b: string[];

    constructor() {
        this.a = [];
        this.b = [];
    }

    addA(v: string) {
        this.a.push(v);
    }

    addB(v: string) {
        this.b.push(v);
    }
}

function main() {
    const t = new TwoArrays();

    // Grow `a` several times; if `a` and `b` alias the same backing
    // store (the bug), this corrupts `b`'s view of its own array.
    t.addA("a1");
    t.addA("a2");
    t.addA("a3");
    t.addA("a4");

    t.addB("b1");

    assert(t.a.length == 4, "a.length should be 4");
    assert(t.b.length == 1, "b.length should be 1 (aliasing bug would corrupt this)");
    assert(t.a[0] == "a1");
    assert(t.a[3] == "a4");
    assert(t.b[0] == "b1");

    print("done.");
}

main();
