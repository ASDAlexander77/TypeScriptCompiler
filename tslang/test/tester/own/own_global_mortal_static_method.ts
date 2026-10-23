// A static method that stores into its class's static (`this.w = x`) is no static constructor: it
// may run any number of times, each store freeing the value before it, so `M.w` stays mortal under
// -mm=own (spec 25) though it is its only store. `I.w`, stored once in its static block, is immortal
// (test-own-global-mortal-static-method reads the IR).
class V {
    constructor(public x: number) {}
}

class M {
    static w: V;
    static set() {
        this.w = new V(3);
    }
}

class I {
    static w: V;
    static {
        I.w = new V(1);
    }
}

function main() {
    M.set();
    print(I.w.x, M.w.x);   // 1 3
}
