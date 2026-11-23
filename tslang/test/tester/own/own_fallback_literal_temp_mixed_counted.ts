// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): from fuzz
// program 1123352, accepted only with the flag. An array literal given straight to a call holds a new object and
// a counted one read out of a global array (`f([mk(i + 4), GA[0]])`): it is given back after the call with what it
// holds - the new object freed, the counted one's count given back while the global keeps it. Before, both leaked
// with the literal every call.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

@noinline function f(p: N[]): number {
    return p[0].x + p[1].x + p.length;
}

const GA: N[] = [mk(7)];

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(i));
    }
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 40; i++) {
        s += f([mk(i + 4), GA[0]]);
        churn();
    }

    print(s, GA[0].x);
    print("done.");
}
