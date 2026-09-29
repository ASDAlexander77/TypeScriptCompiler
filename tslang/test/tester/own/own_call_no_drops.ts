// -mm=own, phase 4: a borrow of a field of `this` or of a parameter survives a call to a function
// that destroys nothing its caller can reach (`__own_no_drops`): a method that reads numbers, a
// helper that builds and fills its own objects. Phase 3 dropped it at any call.
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

class S {
    c: C = new C(0);
    w: number = 2;
    area() {
        return this.w * 2;
    }

    total() {
        const c = this.c;
        const a = this.area();
        churn();
        return c.x + c.v.length + a;
    }
}

function sum(s: S) {
    const c = s.c;
    churn();
    return c.x + s.area();
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const s = new S();
        s.c = new C(i % 10);
        s.c.v.push(1);
        t += s.total() + sum(s);
    }

    assert(t == 1800000);
    print("done.");
}
