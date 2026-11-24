// -mm=own --own-fallback=count: an interface-typed parameter given to two `Shape | null` fields (`h.s = s; k.s =
// s`) is counted only when the module's facts say no class here holds a closure field - a fact
// OwnershipModuleFactsPass caches for the inference (DescriptorObjectsAnalysis). Read without the cache the answer
// is "may hold", and the program is refused instead (own_fallback_cached_analysis_held: the same with a class with a
// closure field, which is refused either way). Pins that the cached analysis is present.
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
