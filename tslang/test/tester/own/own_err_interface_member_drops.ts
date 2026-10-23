// -mm=own rejects: one implementation of a closed interface overwrites a field of its argument, so a
// call through the interface may destroy what `c` borrows (spec 31).
class V {
    constructor(public x: number) {}
}
class H { c: V = new V(5); }
interface Op { run(h: H): number; }
class OA implements Op { run(h: H): number { return h.c.x; } }
class OB implements Op { run(h: H): number { h.c = new V(9); return 0; } }
function f(o: Op, h: H) {
    const c = h.c;
    o.run(h);
    return c.x;
}
function main() {
    const h = new H();
    print(f(new OA(), h), f(new OB(), h));
}
