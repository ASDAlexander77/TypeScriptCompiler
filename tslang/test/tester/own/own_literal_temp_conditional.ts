// -mm=own: an element of an array literal a conditional expression's branch yields (`c ? [mk(i)][0] : null`)
// borrows the literal, as out of a local's array; the literal is held past the conditional and given back at the
// end of the block the conditional is in, with what it holds. It leaked every evaluation before. Also both
// branches, a conditional inside another's branch, strings, a `return` before the end.
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

@noinline function early(i: number): number {
    const o: N | null = i % 2 == 0 ? [mk(i)][0] : null;
    if (!o) {
        return 0;
    }

    return o.x;
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

function main() {
    let total: number = 0;
    let text: number = 0;
    for (let i = 0; i < 40; i++) {
        const o: N | null = i % 2 == 0 ? [mk(i)][0] : null;
        if (o) {
            total += o.x;
        }

        const both = i % 2 == 0 ? [mk(i)][0] : [mk(2), mk(3)][1];
        total += both.x;

        const inner: N | null = i % 2 == 0 ? (i % 3 == 0 ? [mk(i)][0] : null) : null;
        if (inner) {
            total += inner.x;
        }

        if (i % 5 == 0) {
            churn();
            continue;
        }

        const s: string | null = i % 2 == 0 ? ["k" + i, "m" + i][1] : null;
        if (s) {
            text += s.length;
        }

        let l: string | null = i % 2 == 0 ? ["k" + i][0] : null;
        if (i % 3 == 0) {
            l = "z" + i;
        }

        if (l) {
            text += l.length;
        }

        total += early(i);
        churn();
    }

    print(total, text);
    assert(total == 1266, "total");
    assert(text == 104, "text");
    print("done.");
}
