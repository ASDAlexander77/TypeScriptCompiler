// -mm=own --own-fallback=count: own_fallback_cached_analysis with a class that has a closure field: the module's
// facts say a class here may hold a closure, so the interface-typed parameter is not counted and stays an error.
interface Shape { v: number; }
class Node { constructor(public v: number) {} }
class Holder { s: Shape | null = null; }
function link(h: Holder, k: Holder, s: Shape) {
    h.s = s;
    k.s = s;
}
function main() {
    let sum: number = 0;
    for (let i = 0; i < 100; i++) {
        const h = new Holder();
        const k = new Holder();
        link(h, k, new Node(i));
        sum += h.s!.v + k.s!.v;
    }
    print(sum);
    print("done.");
}
class Cl { f: () => number = () => 1; } function useCl() { const c = new Cl(); return c.f(); }
