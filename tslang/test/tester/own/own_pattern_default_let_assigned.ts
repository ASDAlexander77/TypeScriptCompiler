// -mm=own: a `let` a pattern binds with a default that makes a new value, then given another value (`let { p = mk(1) }
// = r; p = mk(3)`), was refused ("'p' borrows 'r.p' and cannot be assigned", "merged from several branches"): its
// value is a merge of a borrow of the field and of the pattern-default slot, both borrows, which own did not see
// as one. It is split as a `let` given a field read is: what it is given later goes into a holder of its own, and the
// `let` borrows that. Also `const [a = mk(1), b = mk(2)] = [mk(i)]` (accepted since the array-default slot). The same
// program under gc and rc.
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
}

@noinline function mk(i: number): N {
    return new N(i);
}

function f(r: R, i: number): number {
    let { p = mk(1) } = r;
    let t = p.x + p.s.length;
    if (i % 3 == 0) {
        p = mk(i + 10);
    }

    return t + p.x + p.s.length;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const r = new R();
        if (i % 2 == 0) {
            r.p = mk(i);
        }

        let { p = mk(1) } = r;
        total += p.x;
        p = mk(3);
        total += p.x + p.s.length;
        total += f(r, i);

        const [a = mk(1), b = mk(2)] = [mk(i)];
        total += a.x + b.x;
    }

    print(total);
    assert(total == 124679, "total");
    print("done.");
}
