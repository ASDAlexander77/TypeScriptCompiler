// A local that walks a list borrows its nodes: the list given up inside the loop frees them, and
// the next step would read a freed node.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    f() {
        let s = 0.0;
        let c = this.head;
        while (c) {
            s += c.v;
            this.head = null;
            c = c.next;
        }

        return s;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
