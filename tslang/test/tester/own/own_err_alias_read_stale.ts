// -mm=own, phase 3 rejects: `x` owns nothing and keeps the first iteration's `h.c`; the next
// iteration reads `h.c` again, but `x` was not assigned on that path, so it still names the value
// the overwrite destroyed.
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
}

function main() {
    const h = new H();
    h.c = new C(5);
    let x: C;
    for (let i = 0; i < 3; i++) {
        const t = h.c;
        if (i == 0) {
            x = t;
        } else {
            print(x.x);
        }

        h.c = new C(i + 10);
        churn();
    }
}
