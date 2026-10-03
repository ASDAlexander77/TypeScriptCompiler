// A module-level object literal keeps what its fields were given (#461). Under rc the call's
// reference was given back at the end of the global's initializer, and nothing else held one, so
// the field pointed at freed memory. Each field is read after allocations of the same size have
// reused anything freed.
class Node {
    v = 0;
    name = "";
}

function mk(name: string) {
    const n = new Node();
    n.name = name;
    return n;
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

let o = { s: mk("made" + 1), n: 7 };
const p = { a: mk("first" + 2), b: mk("second" + 3) };
let t = { inner: { s: mk("inner" + 4) } };

function main() {
    assert(churn() == 1000);
    assert(o.s.name == "made1" && o.n == 7, "a global literal keeps its field's object");
    assert(p.a.name == "first2" && p.b.name == "second3", "a const global literal keeps both fields");
    assert(t.inner.s.name == "inner4", "a nested global literal keeps its object");

    o = { s: mk("next" + 5), n: 8 };
    assert(churn() == 1000);
    assert(o.s.name == "next5" && o.n == 8, "a reassigned global literal keeps the new object");

    print("done.");
}
