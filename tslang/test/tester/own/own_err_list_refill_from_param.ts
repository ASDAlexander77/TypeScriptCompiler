// `n.next = this.head; this.head = n.next;` moves the head into `n` and back when `n` is a node this
// function made (it is freed with nothing in it). A node of the caller's lives on: the list cannot
// take back what the node keeps.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    f(n: Node) {
        n.next = this.head;
        this.head = n.next;
        return 1;
    }
}

function main() { const l = new List(); l.head = new Node(1); const n = new Node(2); print(l.f(n), n.v); }
