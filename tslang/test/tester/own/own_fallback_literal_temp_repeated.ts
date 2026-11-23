// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): the same
// local given twice to a literal beside a new value (`f([mk(2), o, o])`). Plain own refuses the second owner; with
// the flag each occurrence keeps its count, and the literal is given back after the call with what it holds - the
// new value freed, each count given back. Before, the literal was left a borrowing array, and it and `mk(2)`
// leaked every call.
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
@noinline function f(p: N[]): number {
    return p[0].x + p[p.length - 1].x + p.length;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        const o = mk(i);
        total += f([mk(2), o, o]);
        total += o.x;
        churn();
    }

    print(total);
    assert(total == 1760, "total");
    print("done.");
}
