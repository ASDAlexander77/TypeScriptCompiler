// -mm=rc: a conditional expression whose branch yields an optional chain out of an array literal it made
// (`c ? [[mk(i)]][0]?.[0] : null`, a chain of two, a record's field, `??`): the chain is a
// conditional of its own, and its conversion to `| null` another; the branch now looks through what they yield
// and retains it for the result before the literal is given back at the end of the branch. Before, the literal
// was released with what it holds and the receiver's retain came too late (a use after free). The same program
// under gc and own. (`c ? [mk(i)][0]?.s : null` and a chain given to a call, which own refuses, are in
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
        const d: N | null = i % 3 == 0 ? mk(7) : null;

        const a = c ? [[mk(i)]][0]?.[0] : null;
        if (a) {
            total += a.x + a.s.length;
        }

        const e = c ? [[mk(i)]][0]?.[0]?.s : null;
        if (e) {
            total += e.length;
        }

        // a chain of four: eight conditionals in the branch, each looked into once
        const h = c ? [[[[mk(i)]]]][0]?.[0]?.[0]?.[0]?.s : null;
        if (h) {
            total += h.length;
        }

        const f = c ? [{ v: mk(i) }][0]?.v : null;
        if (f) {
            total += f.x + f.s.length;
        }

        const g = d ?? [[mk(i)]][0]?.[0];
        if (g) {
            total += g.x + g.s.length;
        }


        churn();
    }

    print(total);
    assert(total == 137845, "total");
    print("done.");
}
