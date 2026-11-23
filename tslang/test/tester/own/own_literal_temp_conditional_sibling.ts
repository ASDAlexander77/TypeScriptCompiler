// -mm=own: the element a conditional expression's branch yields out of a literal (`c ? [mk(i)][0] : null`), read
// on by a later conditional of the same block (`d ? o : null`, `o ? o : null`, one in an `if`), and a conditional
// six deep: the literal is held to the end of the block the outermost conditional is in, and given back there. It
// leaked every evaluation before.
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

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(1000 + i));
    }
}
function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        const o: N | null = i % 2 == 0 ? [mk(i)][0] : null;
        const q: N | null = i % 3 == 0 ? o : null;
        if (q) {
            total += q.x;
        }

        const r: N | null = o ? o : null;
        if (r) {
            total += r.x;
        }

        if (o) {
            const w: N | null = i % 5 == 0 ? o : null;
            if (w) {
                total += w.x;
            }
        }

        const deep: N | null = i % 2 == 0 ? (i % 3 == 0 ? (i % 4 == 0 ? (i % 5 == 0 ? (i % 6 == 0 ? (i % 7 == 0 ? [mk(i)][0] : null) : null) : null) : null) : null) : null;
        if (deep) {
            total += deep.x;
        }

        churn();
    }

    print(total);
    assert(total == 566, "total");
    print("done.");
}
