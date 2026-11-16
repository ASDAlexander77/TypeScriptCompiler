// A `let` declared in a `try` is declared without a value (the cleanup releases it) and given its
// value by a store: it borrows as one declared outside a `try` does - a parameter, a class read back
// out of `any`, a field, or a local that is still used - and takes a fresh value as its owner.
// Run in a loop with values made at run time, so a double release or a leak shows.
class Node {
    items: string[] = [];
    constructor(public name: string) {
        this.items.push(name + "!");
    }
}

class Holder {
    node = new Node("h");
}

function fromParam(s: Node) {
    try {
        let t = s;
        return t.items.length;
    } catch {
        return 0;
    }
}

function fromAny(n: any) {
    try {
        let back = <Node>n;
        return back.name.length;
    } catch {
        return 0;
    }
}

function fromField(h: Holder) {
    try {
        let n = h.node;
        return n.items[0].length;
    } catch {
        return 0;
    }
}

function fromLocal(i: number) {
    const owner = new Node("o" + i);
    let total = 0;
    try {
        let t = owner;
        total += t.name.length;
    } finally {
        total += owner.items.length;
    }

    return total;
}

function fresh(i: number) {
    try {
        let t = new Node("f" + i);
        return t.items[0].length;
    } catch {
        return 0;
    }
}

function throwsInside(s: Node) {
    let caught = false;
    try {
        let t = s;
        if (t.items.length > 0) {
            throw "x";
        }
    } catch {
        caught = true;
    }

    return caught;
}

function main() {
    const h = new Holder();
    let total = 0;
    for (let i = 0; i < 2000; i++) {
        const n = new Node("n" + i);
        const a: any = new Node("a" + i);
        total += fromParam(n) + fromAny(a) + fromField(h) + fromLocal(i) + fresh(i);
        assert(throwsInside(n), "caught");
        assert(n.items[0] == "n" + i + "!", "n alive");
    }

    assert(total > 0, "total");
    print("done.");
}
