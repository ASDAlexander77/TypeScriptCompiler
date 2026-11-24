// -mm=rc, -mm=own: a pattern's default that is a local (`const e = mk(i + 1); const { p = e } = r`): the default hands
// the local on as the value-or-default's result, which only a conditional's branch was known to do, so the local's
// release took that for an escape and the local leaked every evaluation (+17.5 MB per 400 runs of this loop's shape,
// main too; own too). The local is now given back at the end of its block. A named record, a `let` local, an array
// pattern; the same program under gc. (A record literal source and the assignment form, which own refuses, are in
// rc_destructuring_default_local_literal.)
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

        const e = mk(i + 1);
        const { p = e } = r;
        total += p.x + p.s.length;

        let f = mk(i + 2);
        const { p: q = f } = r;
        total += q.x;

        const g = mk(i + 3);
        const [a = g] = c ? [mk(i)] : [];
        total += a.x;

        churn();
    }

    print(total);
    assert(total == 242092, "total");
    print("done.");
}
