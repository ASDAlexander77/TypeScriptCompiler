// A Shared<T> handle read out of a global and given to a call that gives the global a new value:
// the parameter is a borrow, so the global's release freed the block the callee still read (rc and
// own alike). The caller now holds a count across such a call - and only such a call, so
// `Shared.count` elsewhere is still the number of handles. Counts are checked where there is one
// (rc, own); gc and none report -1. A handle read out of a global's field, which own rejects, is in
// 00shared_global_field_argument.ts.
class Node {
    v = 0;
    tag = "";
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

let bare = new Shared(mk("bare", 2));

// assigns the global it was given
function run(h: Shared<Node>, v: number) {
    bare = new Shared(mk("replaced", 5));
    assert(churn() == 1000);
    assert(countIs(h, 1), "the caller's hold is the only count left");
    return h.value.v == v;
}

// assigns it through a callee
function replaceBare() {
    bare = new Shared(mk("again", 6));
}

function runThrough(h: Shared<Node>) {
    replaceBare();
    assert(churn() == 1000);
    return h.value.v == 5 && h.value.tag == "replaced";
}

function main() {
    assert(countIs(bare, 1), "a call that does not assign the global takes no count");
    assert(run(bare, 2), "a handle given to a call that assigns its global");
    assert(runThrough(bare), "a handle given to a call that assigns its global through a callee");

    let ok = false;
    try {
        ok = run(bare, 6);
    } catch (e) {
        print("caught");
    }

    assert(ok, "a handle given in a try body");
    assert(countIs(bare, 1) && bare.value.tag == "replaced", "the global's own count after the calls");
    print("done.");
}
