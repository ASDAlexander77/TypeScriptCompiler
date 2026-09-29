// -mm=rc: a getter hands back a reference like any other function, so a getter read is settled
// like a call - a temporary is released at the end of its block, a receiver takes it over. Before,
// `h.cc.x` kept the getter's reference and nothing gave it back. Every shape of getter read is
// here, each read after a `churn()` that would reuse a block freed too early: a class getter
// returning a field and a fresh object, an override, a static getter, `super`'s, an interface
// property implemented by a getter, one taken by a `let`, passed to a call, and assigned through.
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

interface HasC {
    get cc(): C;
}

class H implements HasC {
    c: C = new C(1);
    get cc() {
        return this.c;
    }

    set cc(v: C) {
        this.c = v;
    }

    get fresh() {
        return new C(2);
    }

    static s: C = new C(3);
    static get sc() {
        return H.s;
    }
}

class K extends H {
    get cc() {
        return super.cc;
    }

    get fresh() {
        return new C(4);
    }
}

function xOf(c: C) {
    return c.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new H();
        const k: H = new K();
        const f: HasC = h;

        t += h.cc.x + h.fresh.x + k.cc.x + k.fresh.x + H.sc.x + f.cc.x;
        churn();

        let b = h.cc;
        const g = k.fresh;
        churn();
        t += b.x + g.x + xOf(h.cc) + xOf(k.fresh);

        h.cc = new C(5);
        h.cc.x = h.cc.x + 1;
        churn();
        t += h.cc.x + h.cc.v.length + b.x;
    }

    assert(t == 29000);
    print("done.");
}
