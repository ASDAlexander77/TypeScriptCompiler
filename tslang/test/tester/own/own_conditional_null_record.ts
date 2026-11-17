// `c ? null : { h: new H(d), n: 5 }` (and with undefined) given to an interface or `Interface | null`: the
// merge of null and an object literal was `null | {h, n}`, which nothing could cast ("Cast from null |
// {h:H, n:number} to Two | null is not supported", every model). Each branch is now cast to what
// receives the `?:` - a return, a declaration, an argument. Each value is read after churn().
class H {
    constructor(public x: number) {}
}

interface Two {
    h: H;
    n: number;
}

function churn() {
    const keep: H[] = [];
    for (let i = 0; i < 50; i++) keep.push(new H(77777));
    return keep.length;
}

function orNull(d: number): Two | null {
    return d == 0 ? null : { h: new H(d), n: 5 };
}

function orUndefined(d: number): Two | undefined {
    return d == 0 ? undefined : { h: new H(d), n: 5 };
}

function take(t: Two | null) {
    return t ? t.h.x + t.n : -1;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 300; i++) {
        const d = i % 3;
        const a = orNull(d);
        churn();
        s += a ? a.h.x + a.n : 0;
        const b = orUndefined(d);
        churn();
        s += b ? b.h.x : 0;
        const r: Two | null = d == 1 ? { h: new H(d), n: 1 } : null;
        churn();
        s += r ? r.n : 0;
        s += take(d == 2 ? { h: new H(d), n: 1 } : null);
    }

    assert(s == 100 * ((1 + 5) + (2 + 5)) + 100 * (1 + 2) + 100 * 1 + (100 * 3 - 200), "every value read after churn");
    print("done.");
}
