// -mm=rc: a conditional expression whose branch yields an element of an array literal of mixed elements -
// a tuple (`c ? [mk(i), "x"][0] : null`), its field, a string element, through `as`, `??`, an optional chain, a
// call argument: the branch retains it for the result before giving back the `mk(i)` it made at the end of the
// branch. Before, the receiver's retain came too late (a use after free). The same program under gc. Not under
// own, which refuses a tuple literal holding a value it owns anywhere but in a local.
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

        const a = c ? [mk(i), "x"][0] : null;
        if (a) {
            total += a.x + a.s.length;
        }

        const b = c ? [mk(i), "x"][0].s : null;
        if (b) {
            total += b.length;
        }

        const e = c ? ([mk(i), "x"][0] as N) : null;
        if (e) {
            total += e.x + e.s.length;
        }

        const f = c ? [mk(i), ms(i)][1] : null;
        if (f) {
            total += f.length;
        }

        const g = d ?? [mk(i), "x"][0];
        total += g.x + g.s.length;

        const h = c ? [mk(i), "x"][0]?.s : null;
        if (h) {
            total += h.length;
        }

        total += take(c ? [mk(i), "x"][0] : null);

        churn();
    }

    print(total);
    assert(total == 179135, "total");
    print("done.");
}
