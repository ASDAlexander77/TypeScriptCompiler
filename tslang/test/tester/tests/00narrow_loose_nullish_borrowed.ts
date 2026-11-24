// A borrowed `N | null | undefined` narrowed by a loose `!= null` / `== null` - an element read into a `const`, a
// for-of variable, a field - is read as `c !== undefined` reads it, a view of the borrow: -mm=own accepted these
// before the loose narrowing (00narrow_loose_nullish) and refused them with it ("'c' borrows an element and cannot
// be stored, returned or captured"). Every model; output as node's.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class H {
    v: N | null | undefined = undefined;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 100; i++) {
        const vs: (N | null | undefined)[] = [new N(i), null, undefined];
        const c = vs[0];
        if (c != null) total += c.x;
        const d = vs[1];
        if (d == null) total += 1;
        for (const e of vs) {
            if (e != null) total += e.x;
            else if (e === undefined) total += 2;
        }

        const h = new H();
        h.v = new N(3);
        if (h.v != null) total += h.v.x;
        total += h.v == null || h.v.x > 2 ? 1 : 0;
    }

    print(total);
    assert(total == 10600, "total");
    print("done.");
}
