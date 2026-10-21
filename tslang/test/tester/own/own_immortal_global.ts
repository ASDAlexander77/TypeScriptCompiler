// A global never given another value after its initializer is immortal under -mm=own: like a
// string literal, a release skips it, so a read of it may be returned, stored into a field, given
// to another global, or merged with a fresh value - the caller "owns" it either way, and its release
// does nothing. The fresh values `g` makes are still freed.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}

class K {
    static w = new V(1);
    static d = K.w;                       // an immortal global from another
    static f() { return K.w; }            // returned
}

class H { v: V; constructor() { this.v = K.w; } }   // stored into a field

function g(n: number) { return n > 1 ? K.w : new V(n); }   // immortal or fresh

function main() {
    const h = new H();
    print(K.f().x, K.d.x, h.v.x, g(2).x, g(0).x);   // 1 1 1 1 0
    let s: number = 0;
    for (let i = 0; i < 100000; i++) { s += g(i % 3).x; }   // fresh ones are freed
    print(s);                                              // 66666
    assert(s == 66666);
    print("done.");
}
