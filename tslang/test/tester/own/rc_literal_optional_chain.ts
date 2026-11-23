// -mm=rc: an optional chain on an array literal (`[mk(i)]?.[0]`, `c ? [mk(i)]?.[0] : null`, `?.s`, `??`, nested,
// a string literal): the literal's truthiness test and the chain's conditionals that hand its element on kept
// holdReadOnlyNewArrays from holding it, and nothing released it - every evaluation leaked the literal and what it
// holds (+18 MB per 400 runs of this loop's shape, main too). It is now given back at the end of its block, and a
// conditional's branch yielding the element retains it first. The same program under gc and own. (A chain given
// to a call, which own refuses, is in rc_conditional_element_field.)
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

        const a = c ? [mk(i)]?.[0] : null;
        if (a) {
            total += a.x + a.s.length;
        }

        const b = [mk(i)]?.[0];
        if (b) {
            total += b.x + b.s.length;
        }

        const e = c ? [mk(i)]?.[0]?.s : null;
        if (e) {
            total += e.length;
        }

        const f = d ?? [mk(i)]?.[0];
        if (f) {
            total += f.x + f.s.length;
        }

        total += [mk(i)]?.[0]?.x ?? 0;

        const g = c ? ["s" + i]?.[0] : null;
        if (g) {
            total += g.length;
        }

        const h = c ? [[mk(i)]]?.[0]?.[0] : null;
        if (h) {
            total += h.x + h.s.length;
        }

        churn();
    }

    print(total);
    assert(total == 298935, "total");
    print("done.");
}
