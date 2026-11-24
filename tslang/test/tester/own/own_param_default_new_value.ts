// -mm=own: a parameter's default that makes a new value (`function f(p: N = mk(1))`, a method's, a constructor's, an
// arrow's, a closure's, one reading another parameter) leaked it every call that took the default (main too; rc was
// fixed by #1015's frame slot). Under own the same slot owns it, empty unless the default is taken, given back when
// the function exits; the parameter borrows it. A default that borrows (a global, another parameter) is not stored.
// The same program under rc and gc.
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

const G = mk(100);

@noinline function f(p: N = mk(1)): number {
    return p.x + p.s.length;
}

@noinline function fromParam(a: N, p: N = mk(a.x + 1)): number {
    return a.x + p.x;
}

@noinline function borrows(e: N, p: N = e, q: N = G): number {
    return p.x + q.x;
}

class K {
    v: number;
    constructor(p: N = mk(2)) {
        this.v = p.x;
    }

    m(p: N = mk(3)): number {
        return p.x;
    }
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 16; i++) {
        junk.push(new N(i));
    }
}

function main() {
    const arrow = (p: N = mk(4)): number => p.x;
    let total: number = 0;
    for (let i = 0; i < 400; i++) {
        const c = i % 2 == 0;
        const base = i;
        const closure = (p: N = mk(base)): number => {
            return p.x;
        };

        total += f(c ? mk(i) : undefined);
        total += fromParam(mk(i), c ? mk(i) : undefined);
        total += borrows(mk(i), c ? mk(i) : undefined, c ? mk(i) : undefined);
        total += new K(c ? mk(i) : undefined).v + new K().m(c ? mk(i) : undefined);
        total += arrow(c ? mk(i) : undefined) + closure(c ? mk(i) : undefined);

        churn();
    }

    print(total);
    assert(total == 541545, "total");
    print("done.");
}
