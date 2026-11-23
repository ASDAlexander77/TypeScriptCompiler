// -mm=own: an array literal no local holds, read by index (`["k" + i, s][1]`), is given back at the end
// of its block, with what it owns; it leaked every evaluation before. A string element a `let` is given
// is a copy (user ruling 11-23), one a global is given is a copy too, an object element read in place
// is a borrow. Each value read out of a released literal is checked after the literal is gone.
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

// what is freed is reused by the next allocations: a string kept in freed memory would read wrong
function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

let G: string = "g";

function main() {
    let total: number = 0;
    let text = "";
    for (let i = 0; i < 50; i++) {
        // kept string element: a copy the `let` owns, alive after the literal is released
        let x: string = ["k" + i, ms(i)][1];
        // an object element read in place: a borrow of the literal's own
        const o: N = [mk(i), mk(i + 1)][1];
        total += o.x + o.s.length;
        // into a global: copied
        G = ["g" + i, "h" + i][0];
        churn();
        text = x + G;
        total += x.length;
    }

    print(total, text, G);
    assert(text == "m49g49", "kept element");
    assert(G == "g49", "global element");
    assert(total == 1556, "total");
    print("done.");
}
