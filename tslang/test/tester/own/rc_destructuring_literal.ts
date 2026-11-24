// -mm=rc, -mm=own: array destructuring with defaults (`const [a = mk(1)] = [mk(i)]`) reads each element of the array
// literal through a conditional that yields it or the default; that conditional handing the element on kept the
// literal from being given back, so every evaluation leaked it and what it holds (main too, under rc and own). It is
// now given back at the end of its block, after the locals took their counts (under own the locals borrow the
// elements, as out of a named array; a string element a `let` is given is a copy). Objects, strings, numbers, a
// nested pattern, a hole, `let`s given new values. The same program under gc.
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

        let [g = mk(1), h] = [mk(i), mk(i + 1)];
        total += g.x + h.x;
        g = mk(2);
        total += g.x;

        let [t = "q"] = ["s" + i];
        total += t.length;
        t = "t" + i;
        total += t.length;

        const [[k = mk(1)] = []] = [[mk(i)]];
        total += k.x;

        churn();
    }

    print(total);
    assert(total == 650540, "total");
    print("done.");
}
