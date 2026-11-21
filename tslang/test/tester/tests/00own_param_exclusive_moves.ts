// A parameter moved on paths that exclude each other - `if (f) a.push(v); else a[0] = v;`, a move followed by
// `return` and another after it, a `T | undefined` narrowed once and moved on two paths - was "'v' is used here
// after its value was moved" under -mm=own: only one move of a parameter was allowed. Each move now empties the
// parameter, so the release at the exit, kept for a path that moves nothing, frees nothing on the others.
class P {
    constructor(public n: string) {}
}

function put(a: P[], f: number, v: P) {
    if (f == 0) {
        a.push(v);
    } else if (f == 1) {
        a[0] = v;
    } else {
        assert(v.n.length > 0);
    }
}

function keep(a: P[], first: boolean, v: P) {
    if (first) {
        a[0] = v;
        return;
    }

    a[1] = v;
}

function store(a: P[], v: P | undefined) {
    if (v === undefined) {
        return;
    }

    if (a.length == 0) {
        a.push(v);
        return;
    }

    a[0] = v;
}

function main() {
    const a: P[] = [];
    put(a, 0, new P("a"));
    put(a, 0, new P("b"));
    for (let i = 0; i < 1000; i++) {
        put(a, i % 3, new P("x" + (i % 10)));
        keep(a, i % 2 == 0, new P("y" + (i % 7)));
        if (a.length > 4) {
            a.length = 2;
        }
    }

    assert(a.length == 3);
    assert(a[0].n == "y4");
    assert(a[1].n == "y5");

    const b: P[] = [];
    store(b, undefined);
    store(b, new P("s1"));
    store(b, new P("s2"));
    assert(b.length == 1 && b[0].n == "s2");

    print("done.");
}
