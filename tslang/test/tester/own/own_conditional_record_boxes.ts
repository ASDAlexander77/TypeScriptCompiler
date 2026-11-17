// -mm=own: a `?:` between object literals that each hold a new object, given as an interface (each
// literal copied into its box after the merge), and a `const` of such a merge of `@value` literals,
// which MLIRGen used to fold into the merge itself, so that each arm freed its new object at once.
// Each value is read after churn() has reused anything freed too early.
class H {
    constructor(public x: number) {}
}

interface Two {
    h: H;
    n: number;
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

function boxed(d: number): Two {
    return d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
}

function boxedConst(d: number) {
    const o: Two = d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
    churn();
    return o.h.x + o.n;
}

function valueConst(d: number) {
    const o: One = d < 0 ? { h: new H(-d), n: 1 } : { h: new H(d), n: 3 };
    churn();
    return o.h.x + o.n;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 100; i++) {
        const d = i % 2 == 0 ? 2 : -2;
        const r = boxed(d);
        churn();
        s += r.h.x + r.n;
        s += boxedConst(d) + valueConst(d);
    }

    assert(s == 100 * 3 * 2 + 50 * 3 * 3 + 50 * 1 * 3, "every value read after churn");
    print("done.");
}
