// An argument read out of a local is lent to the call, and a later argument may give that local a
// new value before the call takes it: `g(n, (n = mk(2), 0))`. Under rc the assignment released the
// block the first argument read, and the callee read freed memory. The same with a field or an
// element of a local, a conditional argument, a call in a try body, a parameter, and a closure that
// assigns a captured local. Each read follows allocations of the same size, so a freed block is
// reused.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

class H {
    v = new V(0);
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function mk(n: number) {
    return new V(n);
}

function g(a: V, b: number) {
    churn();
    assert(a.tag == "v" + a.x, "the argument's block");
    return a.x * 10 + b;
}

function g2(a: V, b: V) {
    churn();
    assert(a.tag == "v" + a.x && b.tag == "v" + b.x, "the arguments' blocks");
    return a.x * 10 + b.x;
}

function gThrows(a: V, b: number): number {
    churn();
    assert(a.tag == "v" + a.x, "the argument's block");
    throw a.x * 10 + b;
}

function local() {
    let n = mk(1);
    assert(g(n, (n = mk(2), 0)) == 10, "a local");
    assert(n.x == 2, "the local's new value");
}

function localTwice() {
    let n = mk(1);
    assert(g2(n, (n = mk(2), n)) == 12, "a local, then its new value");
}

function localAssignment() {
    let n = mk(1);
    assert(g2(n, n = mk(3)) == 13, "a local, assigned in the next argument");
}

function inLoop() {
    let s = 0;
    for (let i = 0; i < 100; i++) {
        let n = mk(i);
        s += g(n, (n = mk(i + 1), 0));
    }

    assert(s == 49500, "a local in a loop");
}

function field() {
    let h = new H();
    h.v = mk(1);
    assert(g(h.v, (h.v = mk(2), 0)) == 10, "a field of a local");
    const k = new H();
    k.v = mk(3);
    assert(g(k.v, (k.v = mk(4), 0)) == 30, "a field of a const");
}

function element() {
    let a = [mk(1)];
    assert(g(a[0], (a[0] = mk(2), 0)) == 10, "an element of a local");
}

function conditional() {
    let n = mk(1);
    const flag = n.x > 0;
    assert(g(n, flag ? (n = mk(2), 0) : 1) == 10, "a conditional argument");
}

function inTry() {
    let n = mk(1);
    try {
        assert(g(n, (n = mk(2), 0)) == 10, "a call in a try body");
    } catch (e) {
        assert(false, "nothing throws");
    }
}

function thrown() {
    let n = mk(1);
    let caught = 0;
    try {
        gThrows(n, (n = mk(2), 0));
    } catch (e) {
        caught = <number>e;
    }

    churn();
    assert(caught == 10 && n.x == 2, "the call throws");
}

function param(p: V) {
    assert(g(p, (p = mk(2), 0)) == 10, "a parameter");
}

function captured() {
    let n = mk(1);
    const set = () => {
        n = mk(2);
        return 0;
    };
    assert(g(n, set()) == 10, "a captured local, assigned by a closure");
    assert(n.x == 2, "the captured local's new value");
}

function main() {
    local();
    localTwice();
    localAssignment();
    inLoop();
    field();
    element();
    conditional();
    inTry();
    thrown();
    param(mk(1));
    captured();
    print("done.");
}
