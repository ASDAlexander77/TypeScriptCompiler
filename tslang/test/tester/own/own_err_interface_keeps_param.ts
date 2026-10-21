// -mm=own rejects: a method that keeps its argument, reached through an interface. The call
// through the interface is not given the method's facts, so the caller would release the value
// the method keeps (a double free). Until facts reach interface calls (G9), the method's body is
// the error.
class V {
    constructor(public x: number) {}
}
interface SinkV { take(v: V): void; }
class KV implements SinkV { vs: V[] = []; take(v: V) { this.vs.push(v); } }
function main() {
    const ks = new KV();
    const s: SinkV = ks;
    s.take(new V(1));
    print(ks.vs[0].x);
}
