// A `const` read out of a field through an interface (`const v = ix.v`), or out of what a Shared
// handle holds (`const v = s.value`), keeps what it read alive when that field is given a new
// value. Folded, the const was the read itself and held no reference of its own: under rc the
// overwrite freed the block it still read (00const_field_read.ts is the class field case).
// Allocations of the same size reuse anything freed before each read.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class N {
    constructor(public v: V) {}
}

interface IX {
    v: V;
}

function churn() {
    let keep: V[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new V(999));
    return keep.length;
}

let GS = new Shared<V>(new V(1));

function interfaceOfLiteral(i: number) {
    const ix: IX = { v: new V(0) };
    ix.v = new V(i);
    const v = ix.v;
    ix.v = new V(9);
    churn();
    assert(v.x == i && v.tag == "v" + i, "a const read through an interface over an object literal");
}

function interfaceOfClass(i: number) {
    const ix: IX = new N(new V(0));
    ix.v = new V(i);
    const v = ix.v;
    ix.v = new V(9);
    churn();
    assert(v.x == i && v.tag == "v" + i, "a const read through an interface over a class");
}

function interfaceInLoop() {
    let ix: IX = { v: new V(1) };
    for (let i = 0; i < 3; i++) {
        const v = ix.v;
        ix.v = new V(i + 10);
        churn();
        assert(v.x == (i == 0 ? 1 : i + 9), "a const read through an interface, in a loop");
    }
}

function sharedValue(i: number) {
    GS.value = new V(i);
    const v = GS.value;
    GS.value = new V(9);
    churn();
    assert(v.x == i && v.tag == "v" + i, "a const read of a handle's value");
}

function sharedHandleReplaced(i: number) {
    GS = new Shared<V>(new V(i));
    const v = GS.value;
    GS = new Shared<V>(new V(9));
    churn();
    assert(v.x == i && v.tag == "v" + i, "a const read of a handle's value, the handle replaced");
}

function main() {
    for (let i = 2; i < 5; i++) {
        interfaceOfLiteral(i);
        interfaceOfClass(i);
        sharedValue(i);
        sharedHandleReplaced(i);
    }

    interfaceInLoop();
    print("done.");
}
