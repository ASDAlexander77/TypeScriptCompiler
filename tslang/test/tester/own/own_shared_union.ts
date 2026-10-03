// Shared<T> (spec 23.2) in a tagged union whose other members own no block: `Shared<T> | number`
// is a handle, counted as rc counts it. A local, a field and an array element each take a handle,
// switch to a number and back; each owner but one is dropped and its memory reused before the
// last is read. Counts are checked where there is one (rc, own); gc and none report -1.
class Node {
    v = 0;
}

class H {
    u: Shared<Node> | number = 0;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function countIs(s: Shared<Node>, expected: number) {
    const c = Shared.count(s);
    return c < 0 || c == expected;
}

function valueOf(u: Shared<Node> | number) {
    if (typeof u == "number") return -1;
    return u.value.v;
}

// a copy of a parameter put into a union local: the copy and the union each hold a count
function inLet(a: Shared<Node>) {
    const b = a;
    let u: Shared<Node> | number = b;
    assert(countIs(a, 4), "a let union holds a count");
    assert(valueOf(u) == 7);
}

// a const union is a handle of its own too (spec 23.1), as a let is
function inConst(a: Shared<Node>) {
    const b = a;
    const u: Shared<Node> | number = b;
    assert(countIs(a, 4), "a const union holds a count");
    assert(valueOf(u) == 7);
}

function inField(a: Shared<Node>) {
    const b = a;
    const h = new H();
    h.u = b;
    assert(countIs(a, 4), "a union field holds a count");
    assert(valueOf(h.u) == 7);
}

// a handle whose only owner, once this returns, is the union
function lastInUnion(v: number) {
    const s = new Shared(new Node());
    s.value.v = v;
    const u: Shared<Node> | number = s;
    return u;
}

function main() {
    const a = new Shared(new Node());
    a.value.v = 7;
    const keep = a;
    assert(countIs(a, 2), "start");

    inLet(a);
    assert(countIs(a, 2), "the let union gave its count back");
    inConst(a);
    assert(countIs(a, 2), "the const union gave its count back");
    inField(a);
    assert(countIs(a, 2), "the field gave its count back");

    // a field switched to a number and back
    for (let i = 0; i < 1000; i++) {
        const b = a;
        const h = new H();
        h.u = b;
        h.u = i;
        assert(valueOf(h.u) == -1);
        h.u = b;
        assert(valueOf(h.u) == 7);
    }

    assert(countIs(a, 2), "every field gave its count back");

    // a local switched to a number and back
    for (let i = 0; i < 1000; i++) {
        const b = a;
        let u: Shared<Node> | number = b;
        u = 1;
        assert(countIs(a, 3), "a union switched to a number gave its count back");
        u = b;
        assert(valueOf(u) == 7);
    }

    assert(countIs(a, 2), "every local gave its count back");

    // an element switched to a number and back
    const list: (Shared<Node> | number)[] = [a, 1];
    assert(countIs(a, 3), "an element holds a count");
    list[0] = 2;
    assert(countIs(a, 2), "the element switched to a number");
    list[1] = a;
    assert(countIs(a, 3), "another element holds it");

    // a union is the last owner: every other handle is dropped and its memory reused
    const only = lastInUnion(9);
    const elements: (Shared<Node> | number)[] = [lastInUnion(10), 3];
    const field = new H();
    field.u = lastInUnion(11);
    assert(churn() == 1000);
    assert(valueOf(only) == 9, "a union local keeps the node");
    assert(valueOf(elements[0]) == 10 && valueOf(elements[1]) == -1, "an element keeps the node");
    assert(valueOf(field.u) == 11, "a field keeps the node");

    print("done.");
}
