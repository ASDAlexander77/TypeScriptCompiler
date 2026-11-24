// -mm=own: a pattern's default that makes a new value (`const { p = mk(1) } = r`, `p = e ?? mk(1)`, `p = [mk(1)][0]`,
// `p = "q" + i`) leaked it every evaluation that ran the default (main too): the pattern's value borrows the field
// or the default, and nothing owned the new value. It is now held by a slot of the scope, given back when the scope
// ends, as a local is (a default that borrows, `p = e`, is not). Named records; the same program under rc and gc.
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

interface R {
    p?: N;
}

interface RS {
    p?: string;
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
        const r: R = c ? { p: mk(i) } : {};
        const rs: RS = c ? { p: "s" + i } : {};
        const e: N | null = i % 4 == 0 ? mk(9) : null;

        const { p = mk(1) } = r;
        total += p.x + p.s.length;

        const { p: q = e ?? mk(2) } = r;
        total += q.x;

        const { p: w = [mk(3)][0] } = r;
        total += w.x + w.s.length;

        const { p: t = "q" + i } = rs;
        total += t.length;

        const { p: u = e ?? undefined } = r;
        if (u) {
            total += u.x;
        }

        churn();
    }

    print(total);
    assert(total == 164180, "total");
    print("done.");
}
