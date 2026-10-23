// -mm=own rejects: a method that keeps its argument, reached through an exported interface. Another
// module may call through the interface without the method's facts, and would release the value the
// method keeps (a double free), so the call is open and the method's body is the error. The same
// program with the interface not exported compiles (own_interface_facts, spec 31).
class V {
    constructor(public x: number) {}
}
export interface SinkV { take(v: V): void; }
class KV implements SinkV { vs: V[] = []; take(v: V) { this.vs.push(v); } }
function main() {
    const ks = new KV();
    const s: SinkV = ks;
    s.take(new V(1));
    print(ks.vs[0].x);
}
