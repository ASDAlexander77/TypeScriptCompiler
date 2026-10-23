// -mm=own rejects: `k.w = M.w` keeps the read of `M.w` in whatever static the vtable slot of `k`
// holds, so no assignment of `M.w` can be seen ending it (spec 25). `M.w` is mortal: it is written
// through the vtable. Only checkGlobalRead's keep backstop catches this shape; every other rule
// lets it through.
class V {
    constructor(public x: number) {}
}

class M {
    static w = new V(2);
}

class D {
    set(k: typeof M) {
        k.w = M.w;
    }
}

function main() {
    print(M.w.x);
}
