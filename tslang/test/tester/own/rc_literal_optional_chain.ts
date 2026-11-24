// -mm=rc, -mm=own: an optional chain on an array literal (`[mk(i)]?.[0]`, `c ? [mk(i)]?.[0] : null`, `??`,
// `?.x ?? 0`, a string literal): the literal's truthiness test and the chain's conditionals that hand its element
// on kept it from being given back, so every evaluation leaked the literal and what it holds (+18 MB per 400 runs
// of this loop's shape, main too, under rc and own). It is now given back at the end of its block (under own held
// past a conditional whose branch yields the element, as for `c ? [mk(i)][0] : null`), and under rc a branch
// yielding the element retains it first; a chain over a nested literal too. The same program under gc. (A chain
// given to a call and a field read on through a second `?.`, which own refuses, are in rc_conditional_element_field.)
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

        const f = d ?? [mk(i)]?.[0];
        if (f) {
            total += f.x + f.s.length;
        }

        total += [mk(i)]?.[0]?.x ?? 0;

        const g = c ? ["s" + i]?.[0] : null;
        if (g) {
            total += g.length;
        }

        // a chain over a nested literal, held past the conditional as the unchained `c ? [[mk(i)]][0][0] : null`
        const l = c ? [[mk(i)]]?.[0]?.[0] : null;
        if (l) {
            total += l.x + l.s.length;
        }

        const m = c ? [[[mk(i)]]]?.[0]?.[0]?.[0] : null;
        if (m) {
            total += m.x + m.s.length;
        }

        churn();
    }

    print(total);
    assert(total == 338735, "total");
    print("done.");
}
