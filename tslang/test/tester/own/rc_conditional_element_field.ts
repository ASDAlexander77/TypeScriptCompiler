// -mm=rc: a conditional expression whose branch yields a field of an object read out of an array literal's element
// (`c ? [mk(i)][0].s : null`, `c ? ([mk(i)][0] as N).s : null`, an optional chain `c ? [mk(i)][0]?.s : null`), or
// an optional chain given to a call (`take(c ? [[mk(i)]][0]?.[0] : null)`): the branch retains it for the
// conditional's result before the literal is given back at the end of the branch (a use after free before). The
// same program under gc. Not under own, which refuses these (a borrow of the element used after the literal may be
// released, or given on).
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

        const f = c ? [mk(i)][0].s : null;
        if (f) {
            total += f.length;
        }

        const g = c ? ([mk(i)][0] as N).s : null;
        if (g) {
            total += g.length;
        }

        const h = c ? [mk(i)][0]?.s : null;
        if (h) {
            total += h.length;
        }

        total += takes(c ? [mk(i)][0]?.s : null);
        total += take(c ? [[mk(i)]][0]?.[0] : null);

        churn();
    }

    print(total);
    assert(total == 43525, "total");
    print("done.");
}
