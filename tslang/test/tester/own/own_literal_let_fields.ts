// -mm=own: an object literal takes the value of a `let` it is given (`return { s, n }`, `const o =
// { name, node }`): the record moves it out of the local, whose own release goes, and the record's
// owner frees it. The strings are made at run time, so a double release or a lost one shows.
class Node {
    constructor(public v: number) {}
}

function churn() {
    const junk: Node[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Node(999));
    }
}

// returned as it is
function returned(i: number) {
    let s = "s" + i;
    return { s, n: i };
}

// into a local the function returns
function throughLocal(i: number) {
    let s = "t" + i;
    let node = new Node(i);
    const o = { s, node };
    return o;
}

// given a new value before: the record takes the last one
function assignedFirst(i: number) {
    let s = "a" + i;
    s = "b" + i;
    return { s, k: i };
}

// read before the record takes it
function readFirst(i: number) {
    let node = new Node(i);
    const before = node.v;
    return { node, before };
}

// a local record, used and dropped here
function local(i: number) {
    let s = "l" + i;
    const o = { s, n: i };
    churn();
    return o.s.length;
}

// on one path only: the other gives the local back itself
function onePath(i: number) {
    let s = "p" + i;
    if (i % 2 == 0) {
        return { s, n: i };
    }

    return { s: "q", n: -1 };
}

function main() {
    let t = 0;
    for (let i = 0; i < 2000; i++) {
        const k = i % 10;
        const r = returned(k);
        churn();
        assert(r.s == "s" + k && r.n == k, "returned");

        const o = throughLocal(k);
        churn();
        assert(o.s == "t" + k && o.node.v == k, "through a local");

        const a = assignedFirst(k);
        churn();
        assert(a.s == "b" + k, "assigned first");

        const f = readFirst(k);
        churn();
        assert(f.node.v == k && f.before == k, "read first");

        t += local(k);

        const p = onePath(k);
        churn();
        assert(k % 2 == 0 ? p.s == "p" + k : p.s == "q", "one path");
    }

    assert(t == 4000, "t");
    print("done.");
}
