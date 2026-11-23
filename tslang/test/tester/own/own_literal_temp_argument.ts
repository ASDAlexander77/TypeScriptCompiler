// -mm=own: an array literal given straight to a call (`f([mk(9), mk(i + 1)])`) is given back after it,
// with what it owns. Its elements are call results, marked owned only once the calls are settled, so the
// literal looked as if it borrowed them and leaked every call before.
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
    return "m" + i;
}

@noinline function sumObjects(p: N[]): number {
    let n: number = 0;
    for (const o of p) {
        n += o.x + o.s.length;
    }

    return n;
}

@noinline function sumText(p: string[]): number {
    let n: number = 0;
    for (const s of p) {
        n += s.length;
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
    for (let i = 0; i < 50; i++) {
        total += sumObjects([mk(9), mk(i + 1)]);
        total += sumText([ms(i), "a" + i]);
        churn();
    }

    print(total);
    assert(total == 50 * 11 + 1416 + 280, "total");
    print("done.");
}
