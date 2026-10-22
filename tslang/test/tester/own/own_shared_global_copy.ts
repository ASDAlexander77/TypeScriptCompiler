// Shared<T> (spec 23.1): a global that copies another global's handle, in its initializer
// (`let k = a`) or later (`k = a`), or that copies a handle out of another global's field, is a
// second handle and holds a count of its own. Without it, giving back the global it was copied
// from freed the payload the copy still read. Each read follows allocations of the same size, so
// a freed payload is reused. Counts are checked where there is one (rc, own); gc and none report -1.
class Node {
    v = 0;
    tag = "";
}

class Holder {
    s: Shared<Node> = new Shared(mk("field", 3));
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.tag = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

function countIs(s: Shared<Node>, expected: number) {
    const c = Shared.count(s);
    return c < 0 || c == expected;
}

function mk(tag: string, v: number): Node {
    const n = new Node();
    n.tag = tag;
    n.v = v;
    return n;
}

let a = new Shared(mk("a", 1));
let k = a;
let b = new Shared(mk("b", 2));
let kb = b;
const h = new Holder();
let kf = h.s;

function main() {
    assert(countIs(a, 2) && countIs(k, 2), "a copied global counts twice");
    a = new Shared(mk("a2", 10));
    churn();
    assert(k.value.v == 1 && k.value.tag == "a", "the copy outlives the global it was copied from");
    assert(countIs(k, 1), "the copy holds the last count");

    kb = b;
    b = new Shared(mk("b2", 20));
    churn();
    assert(kb.value.v == 2 && kb.value.tag == "b", "a copy assigned again outlives its source");

    h.s = new Shared(mk("field2", 30));
    churn();
    assert(kf.value.v == 3 && kf.value.tag == "field", "a copy of a field outlives the field's overwrite");

    print("done.");
}
