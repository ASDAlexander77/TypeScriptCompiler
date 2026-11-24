// -mm=rc: a conditional expression whose branch turns an optional made in the branch into `| null` - a nested
// conditional's owned result (`c ? (b ? [mk(i)]?.[0] : undefined) : null`, `c ? (b ? [mk(i)][0] : undefined) :
// null`), or an owned optional a call returns (`c ? mo(i) : null`) - reads the value out of that temporary with
// another conditional, and gives the temporary back at the end of the branch. The branch now retains the value it
// yields first; before, the receiver's retain came too late (a use after free or a crash; the chained-literal shape
// was right on main only because the literal leaked). The same program under gc. Not under own, which refuses these
// (a borrow of an element stored past the conditional).
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

@noinline function mo(i: number): N | undefined {
    return i % 8 == 0 ? undefined : mk(i);
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
        const b = i % 4 == 0;

        // chained literal in a nested conditional, other branch `undefined`
        const a = c ? (b ? [mk(i)]?.[0] : undefined) : null;
        if (a) {
            total += a.x + a.s.length;
        }

        // ...other branch an owned value
        const e = c ? (b ? [mk(i)]?.[0] : mk(i)) : null;
        if (e) {
            total += e.x + e.s.length;
        }

        const f = c ? (b ? mk(i) : [mk(i)]?.[0]) : null;
        if (f) {
            total += f.x + f.s.length;
        }

        const g = c ? (b ? [[mk(i)]][0]?.[0] : [mk(i)]?.[0]) : null;
        if (g) {
            total += g.x + g.s.length;
        }

        // an element, a nested element, a tuple element, a record field
        const h = c ? (b ? [mk(i)][0] : undefined) : null;
        if (h) {
            total += h.x + h.s.length;
        }

        const k = c ? (b ? [[mk(i)]][0][0] : undefined) : null;
        if (k) {
            total += k.x + k.s.length;
        }

        const l = c ? (b ? [mk(i), "x"][0] : undefined) : null;
        if (l) {
            total += l.x + l.s.length;
        }

        const m = c ? (b ? [{ v: mk(i) }][0].v : undefined) : null;
        if (m) {
            total += m.x + m.s.length;
        }

        const p = c ? (b ? [mk(i)][0] : [mk(i)]?.[0]) : null;
        if (p) {
            total += p.x + p.s.length;
        }

        // an owned optional a call returns, turned into `| null`
        const q = c ? mo(i) : null;
        if (q) {
            total += q.x + q.s.length;
        }

        const r = c ? (b ? mo(i) : mk(i)) : null;
        if (r) {
            total += r.x + r.s.length;
        }

        const u = c ? (b ? mo(i) : undefined) : null;
        if (u) {
            total += u.x + u.s.length;
        }

        const w = c ? mo(i) ?? null : null;
        if (w) {
            total += w.x + w.s.length;
        }

        churn();
    }

    print(total);
    assert(total == 364907, "total");
    print("done.");
}
