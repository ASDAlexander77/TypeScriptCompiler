// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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
