// -mm=own (spec 33): a parameter kept on some paths only is released on the others. `keepIf` keeps
// `c` only when `f` is true; `put` keeps it under two conditions, returning on each path. It was an
// error before (own_err_param_kept_some_paths): there was nothing to release it with.
class C {
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function keepIf(h: H, c: C, f: boolean) {
    if (f) {
        h.c = c;
    }
}

function put(h: H, c: C, found: boolean, over: boolean): boolean {
    if (found) {
        if (over) {
            h.c = c;
            return true;
        }

        return false;
    }

    return true;
}

function main() {
    const h = new H();
    keepIf(h, new C(1), true);
    keepIf(h, new C(2), false);
    assert(h.c!.x == 1);

    const g = new H();
    assert(put(g, new C(3), true, true));
    assert(!put(g, new C(4), true, false));
    assert(put(g, new C(5), false, true));
    assert(g.c!.x == 3);

    print("done.");
}
