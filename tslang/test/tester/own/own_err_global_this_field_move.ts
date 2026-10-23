// -mm=own rejects: `GK.run()` runs with `this` borrowed from `GK`; `run` moves `c` into a field of
// `this`, then gives `GK` a new value, which frees the old object and `c` with it before `c.x` is
// read.
class C {
    x = 5;
}

class K {
    f: C | null = null;

    run() {
        const c = new C();
        this.f = c;
        GK = new K();
        print(c.x);
    }
}

let GK = new K();

function main() {
    GK.run();
}
