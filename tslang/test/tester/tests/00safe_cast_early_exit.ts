// `if (x === null) return;` narrows x for the rest of the block, as an else branch would. A narrowed
// field is read from the field every time, so an assignment to it writes the field (it was "saving
// to constant object") and a read after the assignment sees the new value (#231).

class N {
    v: number;
    next: N | null = null;

    constructor(v: number) {
        this.v = v;
    }
}

class List {
    head: N | null = null;
    count = 0;

    first(): number {
        if (this.head === null) return -1;
        return this.head.v;
    }

    add(v: number) {
        const n = new N(v);
        n.next = this.head;
        this.head = n;
        this.count++;
    }

    remove(v: number): void {
        if (this.head === null) {
            return;
        }

        if (this.head.v === v) {
            this.head = this.head.next;
            this.count--;
            return;
        }

        let cur = this.head;
        while (cur.next !== null) {
            if (cur.next.v === v) {
                cur.next = cur.next.next;
                this.count--;
                return;
            }

            cur = cur.next;
        }
    }

    replaceHead(v: number): number {
        if (this.head !== null) {
            this.head = new N(v);
            // the field as it is now, not the value the test narrowed
            return this.head.v;
        }

        return -1;
    }
}

function valueOf(h: N | null): number {
    if (h === null) {
        throw "no value";
    }

    return h.v;
}

function sum(items: (N | null)[]): number {
    let s = 0;
    for (const i of items) {
        if (i === null) continue;
        s += i.v;
    }

    return s;
}

function main() {
    const l = new List();
    assert(l.first() == -1);
    l.remove(3);

    l.add(1);
    l.add(2);
    l.add(3);
    assert(l.first() == 3);

    l.remove(3);
    assert(l.first() == 2 && l.count == 2);
    l.remove(1);
    assert(l.first() == 2 && l.count == 1);

    assert(l.replaceHead(9) == 9 && l.first() == 9);

    assert(valueOf(new N(4)) == 4);
    assert(sum([new N(1), null, new N(5)]) == 6);

    print("done.");
}
