// -mm=rc: a conditional expression whose branch yields an element read out of an array literal nested in
// another (`c ? [[mk(i)]][0][0] : null`) - the branch now retains the element for the conditional's result
// before the outer literal is given back at the end of the branch, as for a flat literal (`c ? [mk(i)][0] :
// null`). Before, the outer literal was released with the inner one and the element still on their counts,
// and the receiver's retain after the conditional came too late: an access violation. Also deeper nesting,
// strings, `??`, `&&`, a call argument and a nested conditional; the same program under gc and own.
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

@noinline function ms(i: number): string {
    return "s" + i;
}

@noinline function take(o: N | null): number {
    return o ? o.x + o.s.length : 0;
}

@noinline function takes(s: string | null): number {
    return s ? s.length : 0;
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 16; i++) {
        junk.push(new N(i));
    }
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 400; i++) {
        const c = i % 2 == 0;
        const d: N | null = i % 3 == 0 ? mk(7) : null;

        const o = c ? [[mk(i)]][0][0] : null;
        if (o) {
            total += o.x + o.s.length;
        }

        const deep = c ? [[[mk(i), mk(1)]]][0][0][1] : null;
        if (deep) {
            total += deep.x + deep.s.length;
        }

        const s = c ? [["s" + i]][0][0] : null;
        if (s) {
            total += s.length;
        }

        const t = c ? [[ms(i)]][0][0] : "k";
        total += t.length;

        const q = d ?? [[mk(i)]][0][0];
        total += q.x + q.s.length;

        const a = c && [[mk(i)]][0][0];
        if (a) {
            total += a.x + a.s.length;
        }

        total += take(c ? [[mk(i)]][0][0] : null);
        total += takes(c ? [[ms(i)]][0][0] : null);

        const r = c ? [[mk(i)], [mk(2)]][i % 4 == 0 ? 0 : 1][0] : null;
        if (r) {
            total += r.x + r.s.length;
        }

        const n = c ? (i % 4 == 0 ? [[mk(i)]][0][0] : null) : null;
        if (n) {
            total += n.x + n.s.length;
        }

        const e = c ? [[mk(i)]][0][0] : d;
        if (e) {
            total += e.x + e.s.length;
        }

        // an `N | null` element narrowed by a cast (`<N>`), nested and flat
        const u = c ? <N>[[<N | null>mk(i)]][0][0] : null;
        if (u) {
            total += u.x + u.s.length;
        }

        const w = c ? <N>[<N | null>mk(i)][0] : null;
        if (w) {
            total += w.x + w.s.length;
        }

        churn();
    }

    print(total);
    assert(total == 342917, "total");
    print("done.");
}
