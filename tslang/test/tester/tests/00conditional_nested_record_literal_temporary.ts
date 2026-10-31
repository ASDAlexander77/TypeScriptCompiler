// 00conditional_record_literal_temporary.ts one level down: each branch of the conditional builds a
// @value literal whose field is another @value literal holding an object the branch makes
// (`{ one: { h: new H(d) } }`). Under rc the branch gave its new object back before the receiver
// after the conditional retained the outer literal, so the object was freed under it (heap
// corruption). The branch now retains the literal it yields when a literal inside it holds a branch
// temporary too. Each value is read after churn() has reused anything freed too early.
class H {
    constructor(public x: number) {}
}

@value
interface One {
    readonly h: H;
    readonly n: number;
}

@value
interface Outer {
    readonly one: One;
    readonly m: number;
}

function churn() {
    const keep: H[] = [];
    for (let i = 0; i < 200; i++) keep.push(new H(77777));
    return keep.length;
}

function pick(d: number): Outer {
    return d < 0 ? { one: { h: new H(-d), n: 1 }, m: 2 } : { one: { h: new H(d), n: 3 }, m: 4 };
}

function maybe(d: number): Outer | null {
    return d < 0 ? null : { one: { h: new H(d), n: 5 }, m: 6 };
}

function main() {
    let s: number = 0;
    for (let i = -2; i < 3; i++) {
        const o = pick(i);
        churn();
        s += o.one.h.x + o.one.n + o.m;

        const p = maybe(i);
        churn();
        if (p) s += p.one.h.x + p.one.n + p.m;
    }

    // pick: h.x 2 1 0 1 2 = 6, n+m 3 3 7 7 7 = 27; maybe (i >= 0): h.x 0 1 2 = 3, n+m 11 * 3 = 33
    assert(s == 6 + 27 + 3 + 33, "sum");
    print("done.");
}
