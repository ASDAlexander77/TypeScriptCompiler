// A Shared<T> handle read out of a global's field and given to a call that gives the field a new
// value, and a value read through a global handle given to a call that replaces it: the caller
// holds a count across the call under rc, so the block the parameter reads stays alive. (Own rejects the call: the argument borrows the field.) Counts are checked where
// there is one (rc); gc and none report -1.
class Node {
    v = 0;
    tag = "";
}

class Holder {
    s = new Shared(mk("field", 7));
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

const holder = new Holder();
const G = new Shared(mk("first", 1));

function runField(h: Shared<Node>) {
    holder.s = new Shared(mk("field2", 8));
    assert(churn() == 1000);
    assert(countIs(h, 1), "the caller's hold is the only count left");
    return h.value.v == 7 && h.value.tag == "field";
}

// a value read through a global handle, given to a call that replaces it
function runValue(n: Node) {
    G.value = mk("second", 2);
    assert(churn() == 1000);
    return n.v == 1 && n.tag == "first";
}

function main() {
    assert(countIs(holder.s, 1), "a call that does not assign the field takes no count");
    assert(runField(holder.s), "a handle given to a call that assigns the field it was read from");
    assert(countIs(holder.s, 1) && holder.s.value.v == 8, "the field's own count after the call");
    assert(runValue(G.value), "a value read through a global handle given to a call that replaces it");
    print("done.");
}
