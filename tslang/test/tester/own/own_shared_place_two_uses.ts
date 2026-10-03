// Shared<T> (spec 23.3, Ruling 7): one read of a handle out of a field with two uses, a store into
// another field, which rc counts, and a read through it, which borrows the field read from. With
// nothing that may drop the field in between, both are legal: the store's release of the field's
// old value cannot free the block read, which already holds the store's count. The positive twin
// of own_err_shared_place_two_uses. Each test then drops the field read from, reuses its memory, and
// reads through the field stored into. Counts are checked where there is one (rc, own); gc and none
// report -1.
class Node {
    v = 0;
    tag = "";
}

class P {
    child: Shared<Node> = new Shared(new Node());
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

function look(s: Shared<Node>) {
    return s.value.v;
}

function mkP(tag: string, v: number) {
    const p = new P();
    p.child.value.tag = tag;
    p.child.value.v = v;
    return p;
}

// `Shared.count` of the assignment: the store holds one count, the field read from the other
function countOfAssignment(p: P, q: P) {
    return Shared.count(q.child = p.child);
}

// the assignment given to a call that drops nothing
function lookAtAssignment(p: P, q: P) {
    return look(q.child = p.child);
}

function storedAndCounted() {
    const p = mkP("first", 1);
    const q = mkP("other", 9);
    const c = countOfAssignment(p, q);
    assert(c < 0 || c == 2, "the store and the field each hold a count");
    p.child = new Shared(new Node());
    assert(churn() == 1000);
    const n = Shared.count(q.child);
    assert(n < 0 || n == 1, "the store's count is the last");
    assert(q.child.value.v == 1 && q.child.value.tag == "first", "the stored block is alive");
}

function storedAndRead() {
    const p = mkP("second", 2);
    const q = mkP("other", 9);
    assert(lookAtAssignment(p, q) == 2, "read through the assignment");
    p.child = new Shared(new Node());
    assert(churn() == 1000);
    const n = Shared.count(q.child);
    assert(n < 0 || n == 1, "the store's count is the last");
    assert(q.child.value.v == 2 && q.child.value.tag == "second", "the stored block is alive");
}

// the same field on both sides: the retain comes before the release of the old value
function sameObject() {
    const p = mkP("same", 3);
    const c = countOfAssignment(p, p);
    assert(c < 0 || c == 1, "a field assigned its own value keeps one count");
    assert(churn() == 1000);
    assert(p.child.value.v == 3 && p.child.value.tag == "same", "the block is alive");
}

function main() {
    storedAndCounted();
    storedAndRead();
    sameObject();
    print("done.");
}
