// -mm=rc: a pattern's default that is a local, from a record literal (`const { p = e } = { p: c ? mk(i) : undefined
// }`) and in the assignment form (`({ p: p2 = e ?? mk(1) } = ...)`, `({ p: p3 = e } = r)`): the local is given back
// at the end of its block (it leaked every evaluation, main too). The same program under gc (own refuses these).
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
        const { p = e } = { p: c ? mk(i) : undefined };
        total += p.x + p.s.length;

        let p2: N = mk(0);
        ({ p: p2 = e ?? mk(1) } = { p: c ? mk(i) : undefined });
        total += p2.x;

        let p3: N = mk(0);
        ({ p: p3 = e } = r);
        total += p3.x;

        churn();
    }

    print(total);
    assert(total == 241492, "total");
    print("done.");
}
