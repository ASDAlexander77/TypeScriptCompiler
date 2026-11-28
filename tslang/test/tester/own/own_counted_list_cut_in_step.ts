// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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
