// An inline record literal made for an owning local: `let a = { item: new Leaf(i) }`. MLIRGen
// builds it in a local a `ts.Constant` starts, stores the fresh value into its field, and reads the
// whole record once into `a`. The fresh value moves into the record, and the record into `a`,
// which releases it at the end of its scope; an assignment gives up the record `a` held.
class Leaf {
    v: number;
    s: string;

    constructor(v: number) {
        this.v = v;
        this.s = "leaf-" + v;
    }
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function build(n: number) {
    let total = 0;
    for (let i = 0; i < n; i++) {
        let a = { item: new Leaf(i) };
        total += a.item.v;
    }

    return total;
}

function main() {
    let a = { item: new Leaf(1) };
    assert(churn() == 1000);
    assert(a.item.v == 1 && a.item.s == "leaf-1");

    // overwritten: the old record, and its leaf, are given up
    a = { item: new Leaf(2) };
    assert(churn() == 1000);
    assert(a.item.v == 2 && a.item.s == "leaf-2");

    let b = { item: new Leaf(3), tag: "b" };
    assert(b.item.s == "leaf-3" && b.tag == "b");

    assert(build(20000) == 199990000);
    print("done.");
}
