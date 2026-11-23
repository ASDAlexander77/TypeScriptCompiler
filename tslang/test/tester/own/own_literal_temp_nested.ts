// -mm=own: an array literal nested in another no local holds (`[[mk(i)], [mk(2)]][0][0]`) moves into it as a
// call's result does, and is given back with it at the end of its block - indexed, read by `.length` or a
// `for...of`, given to a call, of strings. The outer literal looked as if it borrowed the inner ones, and every
// evaluation leaked both and what they held before.
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

@noinline function sumNested(p: N[][]): number {
    let n: number = 0;
    for (const a of p) {
        for (const o of a) {
            n += o.x;
        }
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
    let text: number = 0;
    for (let i = 0; i < 40; i++) {
        const v = [[mk(i)], [mk(2)]][0][0];
        total += v.x;
        total += [[mk(i), mk(2)], [mk(3)]][0][1].x;
        total += [[mk(i)]].length;
        total += sumNested([[mk(i)], [mk(1), mk(2)]]);
        for (const a of [[mk(i)], [mk(2)]]) {
            total += a.length;
        }

        total += [[[mk(i)]], [[mk(2)]]][0][0][0].x;
        text += [["s" + i]][0][0].length;
        const s: string = [["s" + i, ms(i)]][0][1];
        text += s.length;
        churn();
    }

    // a `let` given a string read out of a nested literal holds a copy of it, and may be given another value
    let st: string = [["s" + 1, ms(7)]][0][1];
    for (let j = 0; j < 3; j++) {
        st = [[ms(j), "a"]][0][0];
    }

    print(total, text, st);
    assert(total == 780 + 80 + 40 + 780 + 120 + 80 + 780, "total");
    assert(text == 110 + 110, "text");
    assert(st == "m2", "st");
    print("done.");
}
