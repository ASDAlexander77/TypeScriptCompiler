// -mm=own, phase 4: a callee that keeps a parameter (`__own_params`) - a function storing it into
// a field, one passing it on to that function, a method storing it into `this`, one pushing it,
// a constructor's parameter property - takes it over: the caller moves the argument in, from a
// temporary or a `let`, and gives nothing back. Every value is read after a `churn()`.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

class H {
    c: C | null = null;
    items: C[] = [];
    set(c: C) {
        this.c = c;
    }
}

class K {
    constructor(public c: C) {}
}

function keep(h: H, c: C) {
    h.c = c;
}

function forward(h: H, c: C) {
    keep(h, c);
}

function add(h: H, c: C) {
    h.items.push(c);
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const h = new H();
        keep(h, new C(i % 10));
        churn();
        t += h.c!.x;

        let a = new C(1);
        a.v.push(1);
        forward(h, a);
        churn();
        t += h.c!.x + h.c!.v.length;

        h.set(new C(2));
        add(h, new C(3));
        const k = new K(new C(4));
        churn();
        t += h.c!.x + h.items[0].x + k.c.x;
    }

    assert(t == 1550000);
    print("done.");
}
