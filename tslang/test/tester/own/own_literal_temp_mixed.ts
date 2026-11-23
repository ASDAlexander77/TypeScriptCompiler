// -mm=own: an array literal no local holds, given a local's object as well as new ones (`g([mk(i + 4), o])`,
// `[mk(i + 4), o][1].x`), or only locals' objects and read (`[o].length`, `[o][0].x`). A literal that does not
// borrow takes every item it is given: the inference moves `o` in (its own release goes), so the literal is the
// one owner left, and is given back at the end of its block with what it holds. Before, it was never given back,
// and the local's object with it. (A literal of locals given straight to a call still borrows them, as before.)
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

@noinline function sum(p: N[]): number {
    let n: number = 0;
    for (const o of p) {
        n += o.x + o.s.length;
    }

    return n;
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        const o = mk(1);
        total += sum([mk(i + 4), o]);

        let l = mk(i);
        total += sum([mk(2), l]);

        const p = mk(i);
        total += [mk(i + 4), p][1].x;

        const q = mk(3);
        total += [q].length;

        const r = mk(i);
        total += [r][0].x;

        const t = mk(1);
        for (const a of [mk(i), t]) {
            total += a.x;
        }

        // borrowed: a literal of locals given to a call
        const u = mk(2);
        const w = mk(i);
        total += sum([u, w]);
        total += u.x + w.x;
        churn();
    }

    print(total);
    assert(total == 6554, "total");
    print("done.");
}
