// -mm=rc: array destructuring with defaults (`const [a = mk(1)] = [mk(i)]`) reads each element of the array literal
// through a conditional that yields it or the default; that conditional handing the element on kept the literal
// from being given back, so every evaluation leaked it and what it holds (main too). It is now given back at the
// end of its block, after the locals took their counts. Objects, strings, numbers, a nested pattern, a hole. The
// same program under gc and own (own leaks these too: a follow-up).
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
    return "s" + i;
}

@noinline function take(o: N | null): number {
    return o ? o.x + o.s.length : 0;
}

@noinline function takes(s: string | null): number {
    return s ? s.length : 0;
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
        const [a = mk(1)] = [mk(i)];
        total += a.x + a.s.length;

        const [p = 1, q = 2] = [i, i + 1];
        total += p + q;

        const [s = "q", u = "x"] = ["s" + i, ms(i)];
        total += s.length + u.length;

        const [[b = mk(1)] = [mk(2)]] = [[mk(i)]];
        total += b.x + b.s.length;

        const [e = mk(1), , f = mk(3)] = [mk(i), mk(2), mk(4)];
        total += e.x + f.x;

        churn();
    }

    print(total);
    assert(total == 406960, "total");
    print("done.");
}
