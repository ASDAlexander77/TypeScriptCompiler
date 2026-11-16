// `this.head = h.next` frees the node `h` names and gives the list its `next`: `h` may not be used
// after it. Read what is needed of the node first (`const v = h.v`).
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    f(): number {
        const h = this.head;
        if (!h) return -1;
        this.head = h.next;
        return h.v;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
