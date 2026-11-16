// A walking local that may be given the head again does not only step down the list: a field of
// what it holds, overwritten, may free a node an earlier step reached (`m`).
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;
    f() {
        let c = this.head;
        let k = -1;
        while (c) {
            const m = c.next;
            c = c.next;
            if (c && c.v == 3) {
                c = this.head;
            }

            if (c) c.next = null;
            if (m) k = m.v;
        }

        return k;
    }
}

function main() { const l = new List(); l.head = new Node(1); print(l.f()); }
