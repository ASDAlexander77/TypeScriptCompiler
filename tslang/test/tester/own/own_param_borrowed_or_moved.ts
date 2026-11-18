// A parameter the body gives a borrowed element on one path and moves into a local on another, its
// last use (fuzz 949146): the local borrows it, and nothing frees the element the array still holds.
// The move made the function keep the argument, the edge where the paths join released what the
// parameter held on the other path - the element - and the next allocation reused it.
class V {
    x: number;
    items: V[];
    constructor(x: number) {
        this.x = x;
        this.items = [];
    }
}

function mv(i: number): V {
    return new V(i);
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(999));
    }
}

function f2(p3: V, p4: V): number {
    if (p4.x > 7) {
        p4 = p3.items[0];
    } else {
        const v8 = p4;
    }

    return p3.x;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 300; i++) {
        const a = mv(i + 9);
        a.items.push(mv(1));
        s += f2(a, mv(i + 3));
        churn();
        assert(a.items[0].x == 1, "the element is still there");
    }

    assert(s == 300 * 9 + (299 * 300) / 2);
    print("done.");
}
