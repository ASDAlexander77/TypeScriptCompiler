// A module-level variable initialised from another one reads that one's value (#446). Each global
// that runs code gets a constructor, all at the same priority, and LLVM leaves the order of equal
// priorities undefined: optimised, `b`'s constructor was folded before `a`'s and read garbage; the
// JIT ran them in reverse.
class C {
    v = 5;
}

function mk(v: number) {
    const c = new C();
    c.v = v;
    return c;
}

let a: number[] = [1, 2];
let b = a;
const c = new C();
const d = c;
const r = mk(7);
let e: C | null = r;
let s = "text" + 1;
let t = s;
let n = a.length + 1;

function main() {
    assert(b.length == 2 && b[1] == 2, "an array global from another");
    assert(d.v == 5, "a class global from another");
    assert(e !== null && e.v == 7, "a union global from another");
    assert(t == "text1", "a string global from another");
    assert(n == 3, "a number computed from another global");

    print("done.");
}
