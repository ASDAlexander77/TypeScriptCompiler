// `n.next = null` releases the field while `n` still holds the node above: the node `nx` borrows is
// freed, and the walk would go on into it. A walking local frees nothing above what it was given only
// from where it holds that value.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List { head: Node | null = null; }

function unlinkAll(l: List) {
    let n = l.head;
    while (n) {
        const nx = n.next;
        n.next = null;
        n = nx;
    }
}

function main() { const l = new List(); l.head = new Node(1); l.head.next = new Node(2); unlinkAll(l); print("done."); }
