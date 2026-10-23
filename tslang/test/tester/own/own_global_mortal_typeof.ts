// A static written through its class's vtable - `k.w = x` with `k: typeof M`, a `ts.VirtualSymbolRef`
// stored through - may be any class's static the slot holds, so no static is immortal under -mm=own
// (spec 25): `M.w` stays mortal, though its static block stores it once. `I`, a module-level global
// in no vtable, is immortal (test-own-global-mortal-typeof reads the IR).
class V {
    constructor(public x: number) {}
}

class M {
    static w = new V(2);
}

const I = new V(1);

class D {
    set(k: typeof M) {
        k.w = I;
    }
}

function main() {
    print(I.x, M.w.x);   // 1 2
}
