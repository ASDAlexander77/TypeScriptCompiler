// -mm=own: a pattern's default that makes a new value, in a `try` body or a `switch` case (`try { const { p = mk(9) }
// = r; }`, `case 1: { const { p = mk(9) } = r; }`, a case without a block), failed to compile: its slot was made in
// the TryOp's or SwitchOp's region and released where that region does not reach ("operand #0 does not dominate this
// use"). It is made in front of the TryOp/SwitchOp now, as an owned local is. In a loop too, and with a throw out of
// the try body. The same program under rc and gc.
class N {
    x: number;
    s: string;
    constructor(x: number) {
        this.x = x;
        this.s = "n" + x;
    }
}

class R {
    p?: N;
    q?: string;
}

@noinline function mk(i: number): N {
    return new N(i);
}

function mr(i: number, c: boolean): R {
    const r = new R();
    if (c) {
        r.p = mk(i);
        r.q = "q" + i;
    }

    return r;
}

function inTry(i: number): number {
    const r = mr(i, i % 2 == 0);
    try {
        const { p = mk(9) } = r;
        if (i % 5 == 4) throw "x";
        return p.x + p.s.length;
    } catch (e) {
        return 100;
    }
}

function inCaseBlock(i: number): number {
    const r = mr(i, i % 3 == 0);
    switch (i % 4) {
        case 1: {
            const { p = mk(7) } = r;
            return p.x;
        }
        case 2: {
            const { q = "d" + i } = r;
            return q.length;
        }
        default:
            return 1;
    }
}

function inCase(i: number): number {
    const r = mr(i, i % 2 == 1);
    let n: number = 0;
    switch (i % 2) {
        case 0:
            const { p = mk(5) } = r;
            n = p.x;
            break;
        case 1:
            n = 2;
            break;
    }

    return n;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        total += inTry(i) + inCaseBlock(i) + inCase(i);

        const r = mr(i, i % 2 == 0);
        try {
            const { p = mk(i + 1) } = r;
            if (i % 7 == 3) throw "y";
            total += p.x;
        } catch (e) {
            total += 1;
        }

        switch (i % 3) {
            case 0: {
                const { p = mk(3) } = r;
                total += p.x;
                break;
            }
            default:
                total += 2;
        }
    }

    print(total);
    assert(total == 77755, "total");
    print("done.");
}
