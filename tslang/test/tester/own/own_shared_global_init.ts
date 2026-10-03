// Shared<T> (spec 23.1) in module-level globals whose initializer reaches the global through a
// view of its type, or comes from a call: `Shared<T> | undefined` from `new Shared(...)`,
// `Shared<T> | null` (`let` and `const`) from a call, an explicit cast of a call's result, and
// `Shared<T> | number`. Each global is born holding one count. Each test checks that count, copies
// the global, drops the global's own handle (or the copy, for the const), reuses the memory, then
// reads through what is left. Counts are checked where there is one (rc, own); gc and none
// report -1.
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

function mkS(tag: string, v: number) {
    return new Shared(mk(tag, v));
}

let optNew: Shared<Node> | undefined = new Shared(mk("optNew", 1));
let nullCall: Shared<Node> | null = mkS("nullCall", 2);
const constCall: Shared<Node> | null = mkS("constCall", 3);
let castCall = <Shared<Node> | null>mkS("castCall", 4);
let optCall: Shared<Node> | undefined = mkS("optCall", 5);
let numCall: Shared<Node> | number = mkS("numCall", 6);

// `Shared<T> | undefined` from `new Shared(...)`: the initializer goes through `ts.OptionalValue`
function optNewGlobal() {
    if (optNew) assert(countIs(optNew, 1), "an optional global from new Shared is born with one count");
    const g = optNew;
    if (g) assert(countIs(g, 2), "copied");
    optNew = undefined;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 1 && g.value.tag == "optNew", "the copy's block is alive");
    }

    assert(g !== undefined && optNew === undefined);
}

// a `let` `Shared<T> | null` from a call: the call's reference goes through the cast
function nullCallGlobal() {
    if (nullCall) assert(countIs(nullCall, 1), "a nullable let global from a call is born with one count");
    const g = nullCall;
    if (g) assert(countIs(g, 2), "copied");
    nullCall = null;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 2 && g.value.tag == "nullCall", "the copy's block is alive");
    }

    assert(g !== null && nullCall === null);
}

// a `const` `Shared<T> | null` from a call: it cannot be dropped, so the copy is, and the const
// still reads its block
function constCallGlobal() {
    if (constCall) assert(countIs(constCall, 1), "a nullable const global from a call is born with one count");
    let x = constCall;
    if (x) assert(countIs(x, 2), "copied");
    x = null;
    assert(churn() == 1000);
    if (constCall) {
        assert(countIs(constCall, 1), "the copy dropped");
        assert(constCall.value.v == 3 && constCall.value.tag == "constCall", "the const global's block is alive");
    }

    assert(constCall !== null && x === null);
}

// an explicit cast of a call's result
function castCallGlobal() {
    if (castCall) assert(countIs(castCall, 1), "a cast call result global is born with one count");
    const g = castCall;
    if (g) assert(countIs(g, 2), "copied");
    castCall = null;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 4 && g.value.tag == "castCall", "the copy's block is alive");
    }

    assert(g !== null && castCall === null);
}

// `Shared<T> | undefined` from a call
function optCallGlobal() {
    if (optCall) assert(countIs(optCall, 1), "an optional global from a call is born with one count");
    const g = optCall;
    if (g) assert(countIs(g, 2), "copied");
    optCall = undefined;
    assert(churn() == 1000);
    if (g) {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 5 && g.value.tag == "optCall", "the copy's block is alive");
    }

    assert(g !== undefined && optCall === undefined);
}

// `Shared<T> | number` from a call
function numCallGlobal() {
    if (typeof numCall != "number") assert(countIs(numCall, 1), "a handle-or-number global from a call is born with one count");
    const g = numCall;
    if (typeof g != "number") assert(countIs(g, 2), "copied");
    numCall = 7;
    assert(churn() == 1000);
    if (typeof g != "number") {
        assert(countIs(g, 1), "the copy is the only handle left");
        assert(g.value.v == 6 && g.value.tag == "numCall", "the copy's block is alive");
    }

    assert(typeof g != "number" && typeof numCall == "number");
}

function main() {
    optNewGlobal();
    nullCallGlobal();
    constCallGlobal();
    castCallGlobal();
    optCallGlobal();
    numCallGlobal();
    print("done.");
}
