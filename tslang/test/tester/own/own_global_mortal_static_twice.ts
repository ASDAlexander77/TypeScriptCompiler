// A static block that stores into a static twice leaves it mortal under -mm=own (spec 25): a read of
// the first value, taken for the immortal block, would outlive the second store, which frees it.
// `I.w`, stored once, is immortal (test-own-global-mortal-static-twice reads the IR).
class V {
    constructor(public x: number) {}
}

class M {
    static w: V;
    static {
        M.w = new V(2);
        M.w = new V(3);
    }
}

class I {
    static w: V;
    static {
        I.w = new V(1);
    }
}

function main() {
    print(I.w.x, M.w.x);   // 1 3
}
