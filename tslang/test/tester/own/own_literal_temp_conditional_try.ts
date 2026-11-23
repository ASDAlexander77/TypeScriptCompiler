// -mm=own: the literal a conditional expression's branch holds past it (`c ? [mk(i), mk(i + 1)][0] : null`), in
// a `try` whose call throws into the same function's catch: given back on the way to the handler as at the end of
// the block, as a named array is. It leaked every caught throw before.
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
@noinline function thr(i: number) {
    if (i % 4 == 0) {
        throw "x";
    }
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 40; i++) {
        try {
            const o: N | null = i % 2 == 0 ? [mk(i), mk(i + 1)][0] : null;
            thr(i);
            if (o) {
                total += o.x;
            }
        } catch (e) {
            total += 1;
        }

        churn();
    }

    print(total);
    assert(total == 210, "total");
    print("done.");
}
