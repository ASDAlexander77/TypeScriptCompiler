// @strict-null false
// A value moved out of an owning local on some paths only (spec 30): the local's release where the
// paths join is placed instead on the paths that did not move it, so each path frees the value
// exactly once - through its new owner where it moved, through the local where it did not.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

class H {
    c: V | null = null;
}

class Holder {
    constructor(public v: V) {}
}

// keeps its argument: a call of it is a move
function keeps(v: V) {
    const h = new Holder(v);
    return h.v.x;
}

let finallies: number = 0;

function mkt(n: number) {
    if (n > 5) {
        throw n;
    }

    return new V(n);
}

// a move on some paths only, then a call that may throw: the edge's release comes before the
// call, and the unwind path, which joins both, frees nothing of `v` (only `best`, if it moved)
function unwinds(c: boolean, n: number) {
    let best: V = undefined;
    let v = mkt(1);
    try {
        if (c) {
            best = v;
        }

        mkt(n);
    } finally {
        finallies++;
    }

    return best != undefined;
}

function main() {
    const vs = [new V(1), new V(3), new V(2)];
    let best: V = undefined;
    let m: number = 0;
    for (let i in vs) {
        let v = V.plus(vs[i], vs[i]);
        if (v.x > m) {
            best = v; // moved into best on this path only
            m = v.x;
        }
    }

    print(best.x); // 6
    assert(best.x == 6);

    // straight-line conditional move, taken
    let w = new V(7);
    let keep: V = undefined;
    if (m > 5) {
        keep = w;
    } else {
        print("not moved");
    }

    print(keep.x); // 7
    assert(keep.x == 7);

    // ...and not taken: the local still owns it, and frees it
    let w2 = new V(8);
    let keep2: V = undefined;
    if (m > 100) {
        keep2 = w2;
    } else {
        print("not moved");
    }

    assert(keep2 == undefined);

    // a churn loop where a freed or double-freed block would show
    let s: number = 0;
    for (let k = 0; k < 100000; k++) {
        let t = new V(k % 3);
        let b: V = undefined;
        s += t.x;
        if (k % 2 == 0) {
            b = t;
            s += t.x; // a read through the new owner (spec 29)
        }

        if (b != undefined) {
            s += b.x;
        }
    }

    print(s); // 99999 + 50000 + 50000
    assert(s == 199999);

    // an assignment over a value that may have moved frees the old value only where it did not
    let r = new V(0);
    let kept: V = undefined;
    for (let k = 0; k < 1000; k++) {
        if (k % 3 == 0) {
            kept = r;
        }

        r = new V(k);
    }

    print(r.x, kept.x); // 999 998
    assert(r.x == 999 && kept.x == 998);

    // moved on every path, then given a new value on some: the path that gave it one releases it
    // before the paths join
    let given: number = 0;
    for (let k = 0; k < 1000; k++) {
        const h = new H();
        let a = new V(k);
        h.c = a;
        if (h.c.x % 2 == 0) {
            a = new V(k + 1);
            given += a.x;
        }
    }

    print(given); // 1 + 3 + ... + 999
    assert(given == 250000);

    // moved into a call on one arm of a `?:` only (the ray tracer's `shade`)
    let taken: number = 0;
    for (let k = 0; k < 1000; k++) {
        let p = new V(k);
        let q = new V(k + 1);
        taken += k % 2 == 0 ? keeps(p) : q.x;
    }

    print(taken); // (0 + 2 + ... + 998) + (2 + 4 + ... + 1000)
    assert(taken == 500000);

    // ...and on the right of `&&`: the edge that skips it carries the `&&`'s value, so the release
    // goes in a block of its own on that edge, which passes the value on
    let anded: number = 0;
    for (let k = 0; k < 1000; k++) {
        let p = new V(k);
        if (k % 2 == 0 && keeps(p) >= 0) {
            anded++;
        }
    }

    print(anded); // 500
    assert(anded == 500);

    let moved: number = 0;
    let thrown: number = 0;
    for (let k = 0; k < 20000; k++) {
        try {
            if (unwinds(k % 2 == 0, k % 4 < 2 ? 1 : 9)) {
                moved++;
            }
        } catch {
            thrown++;
        }
    }

    print(moved, thrown, finallies); // 5000 10000 20000
    assert(moved == 5000 && thrown == 10000 && finallies == 20000);

    print("done.");
}
