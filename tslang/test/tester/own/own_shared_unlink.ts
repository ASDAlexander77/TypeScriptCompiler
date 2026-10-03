// Shared<T> (spec 23.1): a const handle copied out of a place, or out of a local, is a handle of its
// own, bare or in a union. Each test drops the only other owner, reuses its memory, then reads
// through the const: unlinking a list node (spec 23.1's Node), taking a parent's child before
// clearing the field, and a `Shared<T> | null` const of a let that is then reassigned. Counts are
// checked where there is one (rc, own); gc and none report -1.
class Node {
    v = 0;
    tag = "";
    next: Shared<Node> | null = null;
}

class Parent {
    child: Shared<Node> | null = null;
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

function mk(tag: string, v: number) {
    const n = new Node();
    n.tag = tag;
    n.v = v;
    return n;
}

function link(head: Shared<Node>) {
    head.value.next = new Shared(mk("second", 2));
}

function fill(p: Parent) {
    p.child = new Shared(mk("child", 3));
}

// the list node unlinked from its head: `nxt` is its only handle
function unlink() {
    const head = new Shared(new Node());
    link(head);
    const nxt = head.value.next;
    if (nxt) assert(countIs(nxt, 2), "the field and the const");
    head.value.next = null;
    assert(churn() == 1000);
    if (nxt) {
        assert(countIs(nxt, 1), "the const is the only handle left");
        assert(nxt.value.v == 2 && nxt.value.tag == "second", "the unlinked node is alive");
    }

    assert(nxt !== null);
}

// a parent's child taken into a const before the field is cleared
function takeChild() {
    const p = new Parent();
    fill(p);
    const c = p.child;
    p.child = null;
    assert(churn() == 1000);
    if (c) {
        assert(countIs(c, 1), "the const is the only handle left");
        assert(c.value.v == 3 && c.value.tag == "child", "the child is alive");
    }

    assert(c !== null);
}

// a `Shared<T> | null` const of a let, which is then given another block
function constOfLet() {
    let a = new Shared(mk("alive", 42));
    const c: Shared<Node> | null = a;
    assert(countIs(a, 2), "the let and the const");
    a = new Shared(mk("other", 7));
    assert(churn() == 1000);
    if (c) {
        assert(countIs(c, 1), "the const is the only handle left");
        assert(c.value.v == 42 && c.value.tag == "alive", "the const's block is alive");
    }

    assert(countIs(a, 1) && a.value.tag == "other");
}

function main() {
    unlink();
    takeChild();
    constOfLet();
    print("done.");
}
