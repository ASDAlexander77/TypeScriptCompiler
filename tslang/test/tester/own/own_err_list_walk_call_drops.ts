// A call that may free the list, made while a local walks it: the local's node may be gone after it.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    clear() { this.head = null; }
    f() {
        let s = 0.0;
        let c = this.head;
        while (c) {
            s += c.v;
            this.clear();
            c = c.next;
        }

        return s;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
