// A field of the node a walking local holds may be overwritten (the node itself stays) - but not
// while something read out of that field is still used: `k` names the node `c.next = null` frees.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    f() {
        let c = this.head;
        if (!c) return -1;
        c = c.next;
        if (!c) return -1;
        const k = c.next;
        c.next = null;
        return k ? k.v : -2;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
