// regression test: narrowing by a condition joined with && or ||.
//
// bugs found and fixed:
// - the else branch of `x === undefined && k` read x as defined: what `x === undefined` being false
//   says was taken for the whole `&&`, which is false also when only `k` is. An undefined x was read
//   as a number there, and `x === undefined` was false;
// - `k && x !== undefined` didn't narrow x: the test stopped at `k`;
// - `||` narrowed nothing: not the else branch (false when both sides are), not what follows
//   `if (x === undefined || k) return;`, not the right side of `x === undefined || x > 1`.

class C {
    v = 7;
    get(): number { return this.v; }
}

// the else branch of `&&` is not narrowed
function andElse(a: number | undefined, k: boolean) {
    if (a === undefined && k) {
        return "then";
    } else {
        const e = a;
        return e === undefined ? "else undefined" : "else " + e;
    }
}

function andElseExits(a: C | undefined, k: boolean) {
    if (a === undefined && k) return "then";
    const e = a;
    return e === undefined ? "after undefined" : "after " + e.get();
}

// both sides of `&&` narrow its then branch
function andRight(a: number | undefined, k: boolean) {
    if (k && a !== undefined) {
        const n: number = a;
        return n * 2;
    }

    return -1;
}

function andBoth(a: number | undefined, b: number | undefined) {
    if (a !== undefined && b !== undefined) {
        return a + b;
    }

    return -1;
}

// `||`: its else branch, what follows when its then branch exits, its right side
function orElse(a: number | undefined, k: boolean) {
    if (a === undefined || k) {
        return -1;
    } else {
        const n: number = a;
        return n * 2;
    }
}

function orExits(a: C | undefined, b: C | undefined) {
    if (a === undefined || b === undefined) return -1;
    return a.get() + b.get();
}

function orRight(a: number | undefined) {
    return a === undefined || a > 1;
}

function orThen(a: number | undefined, k: boolean) {
    if (a !== undefined || k) {
        const t = a;
        return t === undefined ? "then undefined" : "then " + t;
    }

    return "else";
}

function orNull(a: C | null, k: boolean) {
    if (a === null || k) return -1;
    return a.get();
}

function orTypeOf(a: string | number, k: boolean) {
    if (typeof a === "string" || k) return -1;
    const n: number = a;
    return n * 2;
}

function ternary(a: number | undefined, k: boolean) {
    return a === undefined || k ? -1 : a * 2;
}

function main() {
    assert(andElse(undefined, false) == "else undefined");
    assert(andElse(3, false) == "else 3");
    assert(andElse(3, true) == "else 3");
    assert(andElse(undefined, true) == "then");

    assert(andElseExits(undefined, false) == "after undefined");
    assert(andElseExits(new C(), false) == "after 7");
    assert(andElseExits(undefined, true) == "then");

    assert(andRight(21, true) == 42);
    assert(andRight(21, false) == -1);
    assert(andRight(undefined, true) == -1);

    assert(andBoth(1, 2) == 3);
    assert(andBoth(1, undefined) == -1);

    assert(orElse(21, false) == 42);
    assert(orElse(21, true) == -1);
    assert(orElse(undefined, false) == -1);

    assert(orExits(new C(), new C()) == 14);
    assert(orExits(new C(), undefined) == -1);
    assert(orExits(undefined, new C()) == -1);

    assert(orRight(undefined));
    assert(orRight(5));
    assert(!orRight(0));

    assert(orThen(4, false) == "then 4");
    assert(orThen(undefined, true) == "then undefined");
    assert(orThen(undefined, false) == "else");

    assert(orNull(new C(), false) == 7);
    assert(orNull(null, false) == -1);
    assert(orNull(new C(), true) == -1);

    assert(orTypeOf(21, false) == 42);
    assert(orTypeOf("s", false) == -1);
    assert(orTypeOf(21, true) == -1);

    assert(ternary(21, false) == 42);
    assert(ternary(undefined, false) == -1);

    print("done.");
}
