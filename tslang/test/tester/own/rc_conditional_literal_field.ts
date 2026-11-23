// -mm=rc: a conditional expression whose branch yields a field read out of an element of an array literal
// (`c ? [mk(i)][0].s : null`, `c ? [{ v: mk(i) }][0].v : null`, a record in a record, an array in a record): the
// branch retains it for the conditional's result before the literal is given back at the end of the branch.
// Before, the literal was released with what it holds and the receiver's retain came too late (a use after
// free the guard-page heap faults at). Under gc the same program; own refuses `[mk(i)][0].s` (a borrow of the
// element's field), so it is not run under own.
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

        // a field read out of an element: of an object, of a record literal, of a record in a record
        const f = c ? [mk(i)][0].s : null;
        if (f) {
            total += f.length;
        }

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

        churn();
    }

    print(total);
    assert(total == 123125, "total");
    print("done.");
}
