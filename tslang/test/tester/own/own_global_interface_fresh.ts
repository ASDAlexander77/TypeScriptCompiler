// An interface-typed global given a fresh class object (`let a: I = mk()`, `let a: I = new V()`)
// owns it. The object reaches the global through a view, `ts.NewInterface`, which hid it from the
// global's initializer: the call's or the `new`'s reference was given back as a discarded
// temporary at the end of the initializer, and the global was born pointing at freed memory
// (own and rc). Each read follows allocations of the same size, so a freed block is reused.
interface I {
    x: number;
    tag: string;
}

class V implements I {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function mk(n: number) {
    return new V(n);
}

let fromCall: I = mk(7);
let fromNew: I = new V(8);
const constFromCall: I = mk(9);

function main() {
    churn();
    assert(fromCall.x == 7 && fromCall.tag == "v7", "a call's result");
    assert(fromNew.x == 8 && fromNew.tag == "v8", "a new object");
    assert(constFromCall.x == 9 && constFromCall.tag == "v9", "a const made by a call");

    fromCall = new V(17);
    fromNew = mk(18);
    churn();
    assert(fromCall.x == 17 && fromCall.tag == "v17", "the call's global reassigned");
    assert(fromNew.x == 18 && fromNew.tag == "v18", "the new object's global reassigned");
    print("done.");
}
