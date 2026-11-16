// The head given to a method of the list itself: the callee reaches the field it is given through
// `this` as well.
class T {
    l: T | null = null;
    constructor(public v: number) {}
}

class L {
    h: T | null = null;
    wrap(a: T | null): T { const n = new T(5); n.l = a; return n; }
    push() { this.h = this.wrap(this.h); }
}

function main() { const l = new L(); l.push(); l.push(); print(l.h ? l.h.v : 0); }
