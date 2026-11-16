// Linked structures built the way a newcomer writes them: a node made with the old head as its `next`
// (`this.head = new Node(v, this.head)`), a tree grown by a function that recurses on a field and gives
// it back (`t.left = insert(t.left, v)`), and walks that assign their parameter (`n = n.next`). Under
// -mm=own the field is given up before the call that keeps its value, and a parameter that only walks
// down from what it was given borrows it.
class Node {
    constructor(public v: number, public next: Node | null) {}
}

class Stack {
    head: Node | null = null;

    push(v: number) {
        this.head = new Node(v, this.head);
    }

    pop(): number {
        if (!this.head) return -1;
        const v = this.head.v;
        this.head = this.head.next;
        return v;
    }

    sum() {
        let s = 0.0;
        for (let c = this.head; c; c = c.next) s += c.v;
        return s;
    }
}

class Tree {
    left: Tree | null = null;
    right: Tree | null = null;
    constructor(public v: number) {}
}

function insert(t: Tree | null, v: number): Tree {
    if (!t) return new Tree(v);
    if (v < t.v) t.left = insert(t.left, v);
    else t.right = insert(t.right, v);
    return t;
}

function treeSum(t: Tree | null): number {
    return t ? t.v + treeSum(t.left) + treeSum(t.right) : 0;
}

function depth(t: Tree | null): number {
    if (!t) return 0;
    const l = depth(t.left);
    const r = depth(t.right);
    return 1 + (l > r ? l : r);
}

// walks that assign their parameter
function len(n: Node | null) {
    let k = 0;
    while (n) {
        k++;
        n = n.next;
    }

    return k;
}

function total(n: Node | null) {
    let s = 0.0;
    while (n) {
        s += n.v;
        n = n.next;
    }

    return s;
}

// drops every node after the first that holds `v`
function cutAfter(n: Node | null, v: number) {
    while (n) {
        if (n.v == v) {
            n.next = null;
            return;
        }

        n = n.next;
    }
}

function make(count: number) {
    const s = new Stack();
    for (let i = 1; i <= count; i++) s.push(i);
    return s;
}

// a node an append-only global keeps: the node made in front of what it holds keeps that too
let KEEP: Node[] = [];

function prepend(n: Node, v: number) {
    n.next = new Node(v, n.next);
}

function main() {
    let checks = 0.0;
    for (let i = 0; i < 2000; i++) {
        const s = make(5);                          // 5 4 3 2 1
        assert(s.sum() == 15 && len(s.head) == 5 && total(s.head) == 15, "push");
        assert(s.pop() == 5 && s.sum() == 10, "pop");

        cutAfter(s.head, 3);                        // 4 3
        assert(len(s.head) == 2 && total(s.head) == 7, "cut");
        assert(len(make(3).head) == 3, "walk a temporary");

        let root: Tree | null = null;
        for (const v of [50, 30, 70, 20, 40, 60, 80, 35]) root = insert(root, v);
        assert(treeSum(root) == 385 && depth(root) == 4, "tree");

        // nodes of the same size made anew: a freed node read by mistake would show here
        const fresh = new Node(1000, null);
        checks += s.sum() + treeSum(root) + fresh.v;
    }

    let kept = 0.0;
    for (let i = 0; i < 300; i++) {
        KEEP.push(new Node(i, null));
        const k = KEEP[i];
        prepend(k, 1);
        prepend(k, 2);                              // i 2 1
        const fresh = new Node(1000, null);
        kept += total(k) + fresh.v;
    }

    assert(checks == 2000 * (7 + 385 + 1000), "checks");
    assert(kept == 300 * 1003 + 299 * 150, "kept");
    print("done.");
}
