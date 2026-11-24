// -mm=own: an array pattern's element default that makes a new value (`const [a = mk(1)] = xs`) is held by a slot of
// the scope, as an object pattern's default is, and the element read borrows that slot or the source's element. Own
// refused these patterns once the default was applied (a new value merged with a borrow of the element had no owner):
// an array source, an array of `T | null` and of `T | undefined`, in a `try` body, a nested pattern of a field, a
// destructured parameter of a function (whose slot is the frame's, given back at exit). The same program under rc
// and gc.
class N {
    x: number;
    s: string;
    constructor(x: number) {
        this.x = x;
        this.s = "n" + x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

class R {
    p?: N;
}

function v(o: N | undefined | null): number {
    return o === undefined ? 1 : o === null ? 2 : o.x + o.s.length;
}

function param([a = mk(1), b = mk(2)]: N[]): number {
    return v(a) + v(b);
}

function paramOptional([a = mk(3)]: (N | undefined)[], i: number): number {
    return v(a) + i;
}

function paramObject({ p = mk(4) }: R): number {
    return v(p);
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const xs: N[] = i % 3 == 0 ? [] : i % 3 == 1 ? [mk(i)] : [mk(i), mk(i + 1)];
        const [a = mk(1), b = mk(2)] = xs;
        total += v(a) + v(b);

        const ns: (N | null)[] = i % 2 == 0 ? [null] : [];
        const [c = mk(5)] = ns;
        total += v(c);

        const us: (N | undefined)[] = i % 2 == 0 ? [undefined, mk(i)] : [mk(i)];
        const [d = mk(6), e = mk(7)] = us;
        total += v(d) + v(e);

        try {
            const [f = mk(8)] = i % 4 == 0 ? [] : [mk(i)];
            if (i % 5 == 4) throw "x";
            total += v(f);
        } catch (err) {
            total += 3;
        }

        const o = { arr: [mk(i)] };
        const { arr: [g = mk(9), h = mk(10)] } = o;
        total += v(g) + v(h);

        const none: N[] = [];
        total += param(xs) + param(none);
        const noneOptional: (N | undefined)[] = [];
        const undefinedOnly: (N | undefined)[] = [undefined];
        total += paramOptional(noneOptional, i) + paramOptional([mk(i)], i) + paramOptional(undefinedOnly, i);

        const r = new R();
        if (i % 2 == 0) r.p = mk(i);
        total += paramObject(r);
    }

    print(total);
    assert(total == 432003, "total");
    print("done.");
}
