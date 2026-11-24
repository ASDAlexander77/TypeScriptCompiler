// -mm=rc: object destructuring with a default (`const { p = mk(1) } = { p: c ? mk(i) : undefined }`): the default
// is made only when the field is undefined, and `ts.OptValueOrDefault` handed back either a borrow of the field or
// that new value, a +1 nobody took - every evaluation that ran the default leaked it (main too). The op now carries
// one reference whichever way it goes (the field's value retained, the default's taken over), and its receiver takes
// it over or it is given back at the end of its block. A record literal, a named record, nested patterns, strings,
// `let`, `??`, and defaults that read a temporary const (`e ?? undefined`, `d!`, `d ?? mk(1)`) or build an array
// literal (`[mk(1)][0]`): those are given back after the default runs, as in a conditional's branch. The same
// program under gc. (Not under own, which refuses a record literal holding a value it owns anywhere but in a local.)
interface R {
    p?: N;
}

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

        const { p = mk(1) } = { p: c ? mk(i) : undefined };
        total += p.x + p.s.length;

        const { q = mk(1), u = "x" } = { q: c ? mk(i) : undefined, u: c ? "s" + i : undefined };
        total += q.x + u.length;

        const r: R = c ? { p: mk(i) } : {};
        const { p: w = mk(2) } = r;
        total += w.x + w.s.length;

        let { p: v = mk(3) } = { p: c ? mk(i) : undefined };
        total += v.x;
        v = mk(4);
        total += v.x;

        const { p: k = mk(5) } = { p: d ?? undefined };
        total += k.x;

        const { p: { p: e = mk(6) } = { p: mk(8) } } = { p: c ? { p: c ? mk(i) : undefined } : undefined };
        total += e.x;

        const { p: f = mk(1) } = { p: mk(i) };
        total += f.x;

        // a default that reads a temporary const, or builds an array literal: given back too
        const e2: N | null = i % 4 == 0 ? mk(9) : null;
        const { p: g = e2 ?? undefined } = { p: c ? mk(i) : undefined };
        if (g) {
            total += g.x;
        }

        const { p: h = d ?? mk(1) } = { p: c ? mk(i) : undefined };
        total += h.x + h.s.length;

        const { p: l = [mk(1)][0] } = { p: c ? mk(i) : undefined };
        total += l.x + l.s.length;

        if (d) {
            const { p: m = d! } = { p: c ? mk(i) : undefined };
            total += m.x;
        }

        churn();
    }

    print(total);
    assert(total == 425130, "total");
    print("done.");
}
