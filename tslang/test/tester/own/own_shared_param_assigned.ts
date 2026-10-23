// A Shared<T> parameter the body gives a new value - in a loop, on one branch, in a try, then
// captured by a closure - and reads afterwards. Under own the parameter's slot only borrowed the new
// handle, which was released at the end of the block that made it, so the read after it saw freed
// memory. The parameter is a counted local of its own, as under rc. Each read follows allocations
// of the same size, so a freed block is reused; the caller's handle keeps its count.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
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

function check(p: Shared<V>, x: number, what: string) {
    assert(p.value.x == x && p.value.tag == "v" + x, what);
}

function inLoop(p: Shared<V>) {
    for (let i = 0; i < 3; i++) {
        p = new Shared(new V(10 + i));
    }
    churn();
    check(p, 12, "assigned in a loop");
    return p.value.x;
}

function onBranch(p: Shared<V>, b: boolean) {
    if (b) {
        p = new Shared(new V(20));
    }
    churn();
    check(p, b ? 20 : 1, "assigned on one branch");
    return p.value.x;
}

function inTry(p: Shared<V>) {
    try {
        p = new Shared(new V(30));
    } finally {
        churn();
    }
    check(p, 30, "assigned in a try");
    return p.value.x;
}

function captured(p: Shared<V>) {
    for (let i = 0; i < 2; i++) {
        p = new Shared(new V(40 + i));
    }
    const f = () => p.value.x;
    churn();
    check(p, 41, "assigned, then captured");
    return f();
}

function main() {
    const s = new Shared(new V(1));
    assert(inLoop(s) == 12, "loop");
    assert(onBranch(s, true) == 20, "branch taken");
    assert(onBranch(s, false) == 1, "branch not taken");
    assert(inTry(s) == 30, "try");
    assert(captured(s) == 41, "closure");
    churn();
    check(s, 1, "the caller's handle");
    const count = Shared.count(s);
    assert(count == -1 || count == 1, "the caller's handle keeps its one count");
    print("done.");
}
