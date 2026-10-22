// A global made by a call that keeps an argument made for it (`new N(new V(4))`, the constructor
// keeping `c`; a function that stores its parameter in what it returns) owns that argument through
// what it was given. Under own the argument used to be released at the end of the global's
// initializer, under the object that kept it. Each read follows allocations of the same size, so a
// freed block is reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class N {
    constructor(public c: V) {}
}

function wrap(v: V) {
    const n = new N(v);
    return n;
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

const n = new N(new V(4));
let w = wrap(new V(5));

function main() {
    churn();
    assert(n.c.x == 4 && n.c.tag == "v4", "a constructor's kept argument");
    assert(w.c.x == 5 && w.c.tag == "v5", "a function's kept argument");
    w = wrap(new V(6));
    churn();
    assert(w.c.x == 6 && n.c.x == 4, "the global given a new value");
    print("done.");
}
