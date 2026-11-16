// The node after the head goes to the list where the head is replaced by it; it cannot go to
// another place too.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    other: Node | null = null;
    f() {
        const h = this.head;
        if (!h) return -1;
        this.other = h.next;
        this.head = h.next;
        return 1;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
