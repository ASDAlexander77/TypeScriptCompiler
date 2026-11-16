// A singly linked list whose nodes own their `next`: walked by a local that steps down it, pushed
// on at the head, appended at the tail, cut, popped and unlinked. Under -mm=own the walking local
// borrows the nodes, and a node taken out of the list is freed where its place is given its `next`.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;

    push(v: number) {
        const n = new Node(v);
        n.next = this.head;
        this.head = n;
    }

    append(v: number) {
        const n = new Node(v);
        let c = this.head;
        if (!c) {
            this.head = n;
            return;
        }

        while (c.next) {
            c = c.next;
        }

        c.next = n;
    }

    sum() {
        let s = 0.0;
        for (let c = this.head; c; c = c.next) s += c.v;
        return s;
    }

    count() {
        let n = 0;
        let c = this.head;
        while (c) {
            n++;
            c = c.next;
        }

        return n;
    }

    find(v: number) {
        let c = this.head;
        while (c) {
            if (c.v == v) return true;
            c = c.next;
        }

        return false;
    }

    // drops every node after the first that holds `v`
    truncate(v: number) {
        let c = this.head;
        while (c && c.v != v) {
            c = c.next;
        }

        if (c) c.next = null;
    }

    pop(): number {
        const h = this.head;
        if (!h) return -1;
        const v = h.v;
        this.head = h.next;
        return v;
    }

    shift(): number {
        if (!this.head) return -1;
        const v = this.head.v;
        this.head = this.head.next;
        return v;
    }

    // unlinks the node after the first that holds `v`
    removeAfter(v: number) {
        let c = this.head;
        while (c) {
            if (c.v == v && c.next) {
                c.next = c.next.next;
                return true;
            }

            c = c.next;
        }

        return false;
    }

    // a new node given the list, which takes it back out of the node at once: the node is freed
    // with nothing in it
    roundTrip() {
        const n = new Node(3);
        n.next = this.head;
        this.head = n.next;
    }

    clear() {
        let n = 0;
        while (this.head) {
            this.head = this.head.next;
            n++;
        }

        return n;
    }
}

function total(l: List) {
    let s = 0.0;
    let c = l.head;
    while (c) {
        s += c.v;
        c = c.next;
    }

    return s;
}

function cut(n: Node) {
    if (n.next) {
        n.next = n.next.next;
    }
}

function main() {
    let checks = 0.0;
    for (let i = 0; i < 2000; i++) {
        const l = new List();
        for (let j = 1; j <= 5; j++) l.push(j);     // 5 4 3 2 1
        l.append(10);
        l.append(20);                               // 5 4 3 2 1 10 20
        assert(l.sum() == 45 && total(l) == 45 && l.count() == 7, "walk");
        assert(l.find(10) && !l.find(7), "find");

        l.truncate(10);                             // 5 4 3 2 1 10
        assert(l.sum() == 25 && l.count() == 6, "truncate");

        assert(l.pop() == 5 && l.shift() == 4, "pop");      // 3 2 1 10
        assert(l.removeAfter(2) && !l.removeAfter(10), "remove");   // 3 2 10
        assert(l.sum() == 15 && l.count() == 3, "after remove");

        if (l.head) cut(l.head);                    // 3 10
        assert(l.sum() == 13, "cut");

        // a local list, walked where it is made
        let local = 0.0;
        for (let c = l.head; c; c = c.next) local += c.v;
        assert(local == 13, "local walk");

        // nodes of the same size made anew: a freed node read by mistake would show here
        const fresh = new Node(1000);
        checks += l.sum() + fresh.v;

        l.roundTrip();
        assert(l.sum() == 13 && l.count() == 2, "round trip");

        assert(l.clear() == 2 && l.count() == 0 && l.pop() == -1, "clear");
    }

    assert(checks == 2026000, "checks");
    print("done.");
}
