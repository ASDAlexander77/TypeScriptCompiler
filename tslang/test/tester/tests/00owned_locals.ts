// Locals that hold a heap reference take one when they are declared and give it back at
// every exit from their scope. Under -mm=rc that is real traffic through __tslang_inc_ref /
// __tslang_dec_ref; under a collector the ops erase and this is an ordinary program. Either
// way the results below must not change, which is what makes it a counting test: a reference
// dropped once too often frees a live value and the reads after it stop matching.

class Node {
    constructor(public v: number) {}
}

// declaration and scope exit, one of each owning shape
function shapes() {
    let s = "abc";
    let a = [1, 2, 3];
    let n = new Node(7);
    let t = ["x", 1];
    let u: string | number = "y";

    assert(s == "abc", "string local");
    assert(a[2] == 3, "array local");
    assert(n.v == 7, "class local");
    assert(t[0] == "x", "tuple local");
    assert(u == "y", "union local");
}

// assignment hands the count over: the incoming value gains this scope as an owner, the
// outgoing one loses it
function reassign() {
    let s = "one";
    s = "two";
    s = s + "!";
    assert(s == "two!", "reassigned string");

    let n = new Node(1);
    n = new Node(2);
    assert(n.v == 2, "reassigned class");
}

// the case retaining first rather than releasing first is there for: releasing the old value
// before the store could free the very value being stored back
function selfAssign() {
    let s = "keep";
    s = s;
    assert(s == "keep", "self-assigned string");

    let n = new Node(3);
    n = n;
    assert(n.v == 3, "self-assigned class");
}

// a local declared inside a loop body is retained and released once per iteration
function loopScope() {
    let total = 0;
    for (let i = 0; i < 4; i++) {
        let s = "ab";
        total = total + s.length;
    }

    assert(total == 8, "per-iteration locals");
}

// break and continue leave the scope early and owe the same releases
function loopExits() {
    let seen = 0;
    for (let i = 0; i < 6; i++) {
        let s = "x";
        if (i == 1) {
            continue;
        }

        if (i == 4) {
            break;
        }

        seen = seen + s.length;
    }

    assert(seen == 3, "break and continue");
}

// returning an owned value must not release the reference the caller is about to receive
function makeName(id: number) {
    let name = "node";
    let n = new Node(id);
    if (id > 0) {
        return name + n.v;
    }

    return name;
}

// a return from a nested block releases every scope it leaves
function nested(flag: boolean) {
    let outer = "out";
    if (flag) {
        let inner = "in";
        return outer + inner;
    }

    return outer;
}

// Paths that reach a local's slot without going through an assignment expression are where a
// missing retain would turn into a release of a reference nobody took, so each one is here on
// purpose rather than for the language feature it names.
function forOf() {
    let names = ["a", "b", "c"];
    let acc = "";
    for (const n of names) {
        acc = acc + n;
    }

    assert(acc == "abc", "for-of binding");
}

function destructure() {
    let pair = ["l", "r"];
    let [x, y] = pair;
    assert(x == "l" && y == "r", "destructured declaration");

    let a = "1";
    let b = "2";
    [a, b] = [b, a];
    assert(a == "2" && b == "1", "destructured assignment");
}

// a captured local outlives the statement that reads it, but the scope's own retain and
// release still pair up
function captured() {
    let s = "cap";
    let f = () => s + "!";
    assert(f() == "cap!", "closure capture");
}

// enough allocation that a block freed one release too early would be handed out again
function churn() {
    let total = 0;
    let widest = 0;
    for (let i = 0; i < 2000; i++) {
        let n = new Node(i);
        let s = "n" + i;
        total = total + n.v;
        if (s.length > widest) {
            widest = s.length;
        }
    }

    assert(total == 1999000, "churn sum");
    assert(widest > 1, "churn strings");
}

function main() {
    shapes();
    reassign();
    selfAssign();
    loopScope();
    loopExits();
    forOf();
    destructure();
    captured();
    churn();

    assert(makeName(5) == "node5", "returned owned value");
    assert(makeName(0) == "node", "returned owned value, other path");
    assert(nested(true) == "outin", "return out of a nested scope");
    assert(nested(false) == "out", "return from the outer scope");

    print("done.");
}
