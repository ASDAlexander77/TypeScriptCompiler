// -mm=own: the read out of an object merged with null (`const e = x.c; const y = c ? x : null`) is used after the
// merge is stored into a field (`h.n = y`), which ends it there: refused.
class N {
    x: number;
    c: N | null = null;
    constructor(x: number) {
        this.x = x;
    }
}

class H {
    n: N | null = null;
}

@noinline function mkc(i: number): N {
    const n = new N(i);
    n.c = new N(i + 100);
    return n;
}

function main() {
    let t: number = 0;
    const h = new H();
    for (let i = 0; i < 20; i++) {
        const x = mkc(i);
        const e = x.c;
        const y: N | null = i % 2 == 0 ? x : null;
        h.n = y;
        if (e) {
            t += e.x;
        }
    }

    print(t);
    print("done.");
}
