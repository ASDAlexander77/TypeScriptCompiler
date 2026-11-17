// -mm=own: a `?:` between `@value` object literals that each hold a new object (`c ? { h: new H(1) } :
// { h: new H(2) }`), or between one and null, moves the literal it chooses into what takes the merge -
// the result, or an owning `let` - with the new object in it. Each value is read after churn() has
// reused anything freed too early.
class H {
    constructor(public x: number) {}
}

@value
interface One {
    readonly h: H;
    readonly n: number;
}

function churn() {
    const keep: H[] = [];
    for (let i = 0; i < 200; i++) keep.push(new H(77777));
    return keep.length;
}

function pick(d: number): One {
    return d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
}

function maybe(d: number): One | null {
    return d < 0 ? null : { h: new H(d), n: 5 };
}

function inLet(d: number) {
    let o: One = d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
    churn();
    return o.h.x + o.n;
}

let G: One | null = null;

// into an array literal, and into a global
function inArray(d: number) {
    const a = [d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 }];
    churn();
    return a[0].h.x;
}

function intoGlobal(d: number) {
    G = d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
}

function main() {
    let s: number = 0;
    for (let i = -2; i < 3; i++) {
        const o = pick(i);
        churn();
        s += o.h.x + o.n;

        const m = maybe(i);
        churn();
        if (m) s += m.h.x + m.n;

        s += inLet(i);
        s += inArray(i);
        intoGlobal(i);
        churn();
        s += G.h.x;
    }

    assert(s == 17 + 18 + 17 + 6 + 6, "sum");
    print("done.");
}
