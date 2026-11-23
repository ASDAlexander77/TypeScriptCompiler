// -mm=own: an array literal no local holds read through `.length`, a `for...of` or a spread is given back
// at the end of its block, with what it owns; each leaked every evaluation before.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

function main() {
    let total: number = 0;
    let text = "";
    const named = ["a" + 1];
    for (let i = 0; i < 30; i++) {
        total += [mk(i), mk(i + 1)].length + ["a" + i].length;
        for (const s of ["p" + i, "q" + i]) {
            text = s;
            total += s.length;
        }

        for (const o of [mk(i), mk(i + 10)]) {
            total += o.x;
        }

        total += [...named, "b" + i].length;
        const k: string = [...named, "c" + i][1];
        churn();
        total += k.length;
    }

    print(total, text);
    assert(text == "q29", "for-of element");
    print("done.");
}
