// `p` names the node `c` held before the step: `p.next = null` frees the node `c` was just given.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List { head: Node | null = null; }

function f(l: List) {
    let s = 0.0;
    let c = l.head;
    while (c) {
        const p = c;
        c = c.next;
        p.next = null;
        if (c) s += c.v;
    }

    return s;
}

function main() { const l = new List(); l.head = new Node(1); l.head.next = new Node(2); print(f(l)); }
