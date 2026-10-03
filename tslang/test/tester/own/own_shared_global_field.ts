// Shared<T> (spec 23.1) in module-level globals initialised from a field or an element read: of
// another global's object (`holder.child`), of a temporary (`new P().child`), of a call's result
// (`mkP().child`), and of a call's array (`mkArr()[0]`). Such a read carries no reference of its
// own, so the global takes one: it is born at count 2 when the field's object lives on, and at
// count 1 when the object or array was a temporary given back at the end of the initializer. Each test drops the other handle, reuses
// the memory, then reads through the global. Counts are checked where there is one (rc, own); gc
// and none report -1.
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

class P {
    child: Shared<Node> = new Shared(mk("child", 9));
}

function mkP(tag: string, v: number) {
    const p = new P();
    p.child.value.tag = tag;
    p.child.value.v = v;
    return p;
}

function mkArr(tag: string, v: number) {
    return [mkP(tag, v).child];
}

const holder = mkP("held", 1);
let fromHeld: Shared<Node> | null = holder.child;
let fromTemp: Shared<Node> = new P().child;
let fromCall: Shared<Node> | null = mkP("called", 3).child;
let fromElem: Shared<Node> | null = mkArr("elem", 4)[0];

// `holder.child`: the field and the global hold one count each
function fromHeldGlobal() {
    if (fromHeld) assert(countIs(fromHeld, 2), "a global from a live object's field takes a count of its own");
    holder.child = new Shared(mk("other", 0));
    assert(churn() == 1000);
    if (fromHeld) {
        assert(countIs(fromHeld, 1), "the global is the only handle left");
        assert(fromHeld.value.v == 1 && fromHeld.value.tag == "held", "the global's block is alive");
    }

    assert(fromHeld !== null);
}

// `new P().child`: the temporary gave its count back; the global's is the only one
function fromTempGlobal() {
    assert(countIs(fromTemp, 1), "a global from a temporary's field is born with one count");
    assert(churn() == 1000);
    assert(countIs(fromTemp, 1), "still one");
    assert(fromTemp.value.v == 9 && fromTemp.value.tag == "child", "the global's block is alive");
}

// `mkP(..).child`: as the temporary, through a call
function fromCallGlobal() {
    if (fromCall) assert(countIs(fromCall, 1), "a global from a call result's field is born with one count");
    const g = fromCall;
    if (g) assert(countIs(g, 2), "copied");
    fromCall = null;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 3 && g.value.tag == "called", "the copy's block is alive");
    }

    assert(g !== null && fromCall === null);
}

// `mkArr(..)[0]`: an element of a temporary array
function fromElemGlobal() {
    if (fromElem) assert(countIs(fromElem, 1), "a global from a temporary array's element is born with one count");
    assert(churn() == 1000);
    if (fromElem) {
        assert(countIs(fromElem, 1), "still one");
        assert(fromElem.value.v == 4 && fromElem.value.tag == "elem", "the global's block is alive");
    }

    assert(fromElem !== null);
}

function main() {
    fromHeldGlobal();
    fromTempGlobal();
    fromCallGlobal();
    fromElemGlobal();
    print("done.");
}
