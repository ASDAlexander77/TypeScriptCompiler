// A value read out of an assigned global, lent to a call that assigns other globals but not that one:
// only a store to the global releases its block, so the value outlives the call. Own refused every
// call that might drop anything; it now refuses one that may assign this global, or any it cannot
// name. Each read follows allocations of the same size, so a freed block would be reused.
class N {
    tag = "";
    constructor(public x: number) {
        this.tag = "n" + x;
    }
}

let G = new N(1);
let H = new N(5);

function churn() {
    const keep: N[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new N(777));
    return keep.length;
}

function setH(x: number) {
    H = new N(x);
}

function other(n: N) {
    H = new N(n.x + 10);
    churn();
    assert(n.tag == "n" + n.x, "the lent value, after the callee assigned another global");
    return n.x;
}

function deeper(n: N) {
    setH(n.x + 20);
    churn();
    assert(n.tag == "n" + n.x, "the lent value, after a callee's callee assigned another global");
    return n.x;
}

function main() {
    G = new N(2);
    assert(other(G) == 2 && H.x == 12, "a callee that assigns another global");
    G = new N(3);
    assert(deeper(G) == 3 && H.x == 23, "a callee's callee that assigns another global");
    print("done.");
}
