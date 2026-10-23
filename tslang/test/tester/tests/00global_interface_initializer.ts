// An interface-typed global initialized with a class object (`let a: I = K.w`, `let a: I = mk()`)
// holds the object through a view, `ts.NewInterface`. Under rc that view hid the object from the
// global's initializer: a read of another global, a field or an element took no count, so giving
// either side a new value freed the block the other still read; and a call's or a `new`'s
// reference was given back as a discarded temporary at the end of the initializer, so the global
// was born pointing at freed memory. Each read follows allocations of the same size, so a freed
// block is reused.
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

class H {
    v = new V(3);
}

class K {
    static w = new V(1);
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

let G = new V(2);
const h = new H();
const arr = [new V(4)];

// the interface global is given a new value: the source keeps its block
let aStatic: I = K.w;
let aGlobal: I = G;
let aField: I = h.v;
let aElement: I = arr[0];

// the source is given a new value: the interface global keeps its block
let G2 = new V(5);
const h2 = new H();
const arr2 = [new V(6)];
let bGlobal: I = G2;
const bConst: I = G2;
let bField: I = h2.v;
let bElement: I = arr2[0];

// a fresh object: the global takes its reference
let cCall: I = mk(7);
let cNew: I = new V(8);

function main() {
    churn();
    assert(cCall.x == 7 && cCall.tag == "v7", "a call's result");
    assert(cNew.x == 8 && cNew.tag == "v8", "a new object");

    aStatic = new V(11);
    aGlobal = new V(12);
    aField = new V(13);
    aElement = new V(14);
    churn();
    assert(K.w.x == 1 && K.w.tag == "v1", "a static field read");
    assert(G.x == 2 && G.tag == "v2", "a global read");
    assert(h.v.x == 3 && h.v.tag == "v3", "a field read");
    assert(arr[0].x == 4 && arr[0].tag == "v4", "an element read");

    G2 = new V(15);
    h2.v = new V(16);
    arr2[0] = new V(17);
    churn();
    assert(bGlobal.x == 5 && bGlobal.tag == "v5", "a global read, the global reassigned");
    assert(bConst.x == 5 && bConst.tag == "v5", "a const of a global, the global reassigned");
    assert(bField.x == 3 && bField.tag == "v3", "a field read, the field reassigned");
    assert(bElement.x == 6 && bElement.tag == "v6", "an element read, the element reassigned");

    cCall = new V(18);
    cNew = new V(19);
    churn();
    assert(cCall.x == 18 && cNew.x == 19, "the fresh objects' globals reassigned");
    assert(aStatic.x == 11 && aGlobal.x == 12 && aField.x == 13 && aElement.x == 14, "the new values");
    print("done.");
}
