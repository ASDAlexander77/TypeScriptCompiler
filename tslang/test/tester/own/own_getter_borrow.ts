// -mm=own, phase 4: a result that borrows an argument (`__own_result_borrows`). A getter, a method
// and a function returning a field of their argument, a method forwarding another's borrow, a
// field or `null`, and a function returning its parameter: the callee gives no reference, the
// caller takes none, and the result is bounded by the argument, read after a `churn()`.
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
    c: C = new C(0);
    get cc() {
        return this.c;
    }

    first() {
        return this.c;
    }

    firstOrNull(k: number): C | null {
        return k > 0 ? this.c : null;
    }
}

class W {
    h: H = new H();
    inner() {
        return this.h.first();
    }
}

function firstOf(h: H) {
    return h.c;
}

function same(c: C) {
    return c;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const h = new H();
        h.c = new C(i % 10);
        h.c.v.push(1);
        const a = firstOf(h);
        let b = h.first();
        churn();
        t += a.x + b.x + h.cc.x + b.v.length;

        const w = new W();
        w.h.c = new C(1);
        const d = w.inner();
        churn();
        t += d.x;

        const n = h.firstOrNull(i % 2);
        if (n) {
            t += n.x;
        }

        let s = same(new C(2));
        churn();
        t += s.x + same(h.c).x;
    }

    assert(t == 2450000);
    print("done.");
}
