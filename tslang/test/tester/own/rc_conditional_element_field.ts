// -mm=rc: a conditional expression whose branch yields a field of an object read out of an array literal's element
// (`c ? [mk(i)][0].s : null`, `c ? ([mk(i)][0] as N).s : null`): the branch retains it for the conditional's
// result before the literal is given back at the end of the branch (a use after free before). The same program
// under gc. Not under own, which refuses both (a borrow of the element used after the literal may be released).
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

        churn();
    }

    print(total);
    assert(total == 1490, "total");
    print("done.");
}
