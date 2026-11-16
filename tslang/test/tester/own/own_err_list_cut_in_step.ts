// The place `c.next` is that of the node `c` holds before the step: `c = c.next` gives `c` the next
// node, and storing into the place then frees it. A walking local frees nothing above the value
// read only where it is read after it was given that value.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List { head: Node | null = null; }

function f(l: List) {
    let s = 0.0;
    let c = l.head;
    while (c) {
        c.next = (c = c.next, null);
        if (c) s += c.v;
    }

    return s;
}

function main() { const l = new List(); l.head = new Node(1); l.head.next = new Node(2); print(f(l)); }
