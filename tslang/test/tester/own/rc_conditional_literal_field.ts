// -mm=rc: a conditional expression whose branch yields a field read out of an element of an array literal
// (`c ? [{ v: mk(i) }][0].v : null`, a record in a record, an array in a record), also past a cast that only
// narrows (`[[<N | null>mk(i)]][0][0]!.s`, `([[<N | null>mk(i)]][0][0] as N).s`): the branch retains it for the
// conditional's result before the literal is given back at the end of the branch. Before, the literal was
// released with what it holds and the receiver's retain came too late (a use after free the guard-page heap
// faults at). The same program under gc and own. (`c ? [mk(i)][0].s : null`, which own refuses, is in
// rc_conditional_element_field.)
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

        // a field read out of an element: of a record literal, of a record in a record, of an array in a record
        const g = c ? [{ v: mk(i) }][0].v : null;
        if (g) {
            total += g.x + g.s.length;
        }

        const h = c ? [{ v: "s" + i }][0].v : null;
        if (h) {
            total += h.length;
        }

        const k = c ? [{ v: { w: mk(i) } }][0].v.w : null;
        if (k) {
            total += k.x + k.s.length;
        }

        const l = c ? [{ v: [mk(i)] }][0].v[0] : null;
        if (l) {
            total += l.x + l.s.length;
        }

        // ...past a cast that only narrows an `N | null` element: `!`, `as N`, a record's field `!`, a flat literal
        const m = c ? [[<N | null>mk(i)]][0][0]!.s : null;
        if (m) {
            total += m.length;
        }

        const p = c ? ([[<N | null>mk(i)]][0][0] as N).s : null;
        if (p) {
            total += p.length;
        }

        const q = c ? [{ v: <N | null>mk(i) }][0].v!.s : null;
        if (q) {
            total += q.length;
        }

        const r = c ? [<N | null>mk(i)][0]!.s : null;
        if (r) {
            total += r.length;
        }

        churn();
    }

    print(total);
    assert(total == 125360, "total");
    print("done.");
}
