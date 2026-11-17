// -mm=own: a `let` given reads of other owning locals, and assigned again (`let b = q; ... b = r`),
// borrows each value until its next assignment: none of them moves into it, and an owner it no
// longer borrows can be moved afterwards. A merge of owning locals (`qq = r && q`) is borrowed too.
class T {
    v: number;
    constructor(v: number) {
        this.v = v;
    }
}

class H {
    t: T | null = null;
}

function reassigned() {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    assert(b.v == 1);
    b = r;
    assert(b.v == 2 && q.v == 1 && r.v == 2);
}

function inLoop() {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    let s: number = 0;
    for (let i = 0; i < 3; i++) {
        s += b.v;
        if (i == 0) b = r;
    }

    assert(s == 5);
}

function onSomePaths(): T {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    if (r.v > 5) b = r;
    assert(b.v == 1);
    return q;
}

function ownerMovedAfter(): T {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    assert(b.v == 1);
    b = r;
    assert(b.v == 2);
    return q;
}

function ownerStoredAfter() {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    assert(b.v == 1);
    b = r;
    const h = new H();
    h.t = q;
    assert(b.v == 2 && h.t.v == 1);
}

function merges() {
    let q = new T(1);
    let r: T | null = null;
    let qq = q && r;
    assert(qq == null);
    qq = r && q;
    assert(qq == null);
    r = new T(4);
    qq = r || q;
    assert(qq.v == 4);
}

class Node {
    constructor(public value: number, public next: Node | undefined) {}
}

// a list walked from a local head: `node` borrows `head`, then each `node.next`
function walkWhile() {
    let head: Node | undefined = new Node(1, new Node(2, new Node(3, undefined)));
    let sum: number = 0;
    let node = head;
    while (node) {
        sum += node.value;
        node = node.next;
    }

    assert(sum == 6 && head.value == 1);
}

function walkFor() {
    let head: Node | undefined = new Node(1, new Node(2, undefined));
    let sum: number = 0;
    for (let node = head; node; node = node.next) {
        if (node.value == 1) node.next = new Node(5, undefined);
        sum += node.value;
    }

    assert(sum == 6);
}

// returned: it owns what it is given, as before
function returned(): T {
    let q = new T(1);
    let r = new T(2);
    let b = q;
    b = r;
    return b;
}

function main() {
    for (let i = 0; i < 1000; i++) {
        reassigned();
        inLoop();
        assert(onSomePaths().v == 1);
        assert(ownerMovedAfter().v == 1);
        ownerStoredAfter();
        merges();
        walkWhile();
        walkFor();
        assert(returned().v == 2);
    }

    print("done.");
}
