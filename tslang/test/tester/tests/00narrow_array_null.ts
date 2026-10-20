// An array or null, made by a conditional expression and narrowed by testing it.
//
// `i < 3 ? ["a"] : null` was a `const_array | null`, and making it the `string[] | null` it is
// returned as called a generated cast function, whose `else` narrowed `t` to a copy of the
// literal as an array: a local that owned it, given back at the end of the function, outside the
// branch it was made in - "operand #0 does not dominate this use", under every model. A literal
// array in a branch of `?:` is now an array there, as TypeScript has it, and the branch hands it
// out with a reference of its own, as `new C()` is, so -mm=own sees the merge as one fresh array
// instead of rejecting it (`either`, which never failed to verify, was rejected by own too).
//
// And `if (a)` of a `string[] | null` did not narrow it, since an array holds null in its own
// pointer and the union has no tag: `a[0]` then indexed the union.

function pick(i: number) {
    return i < 3 ? ["a"] : null;
}

function picked(i: number): string[] | null {
    return i < 3 ? ["a", "b"] : null;
}

function either(i: number): string[] {
    return i < 3 ? ["a"] : ["b", "c"];
}

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

function made(i: number): string[] | null {
    if (i < 3) return ["c"];
    return null;
}

function main() {
    const a = pick(1);
    const b = pick(5);
    assert(a != null && b == null);
    if (a) assert(a[0] == "a");

    const c = picked(1);
    if (c) {
        assert(c.length == 2 && c[1] == "b");
    }

    assert(picked(9) == null);

    const g = either(1);
    const h = either(7);
    churn();
    assert(g.length == 1 && g[0] == "a" && h.length == 2 && h[1] == "c");
    g.push("z");
    assert(g.length == 2 && h.length == 2);

    const d = made(1);
    if (!d) assert(false);
    else assert(d[0] == "c");

    let count = 0;
    let e = made(2);
    while (e) {
        count += e.length;
        e = null;
    }

    for (let f = made(0); f; f = null) {
        count += f.length;
    }

    assert(count == 2);
    print("done.");
}
