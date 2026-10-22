// -mm=own rejects: a method that keeps its argument, reached through an exported interface. Another
// module may call through the interface without the method's facts, and would release the literal
// the method keeps (a double free), so the call is open and the method's body is the error. The same
// program with the interface not exported compiles (own_interface_facts, spec 31).
class V {
    constructor(public x: number) {}
}
interface Ray { start: V; dir: V; }
export interface Sink { take(r: Ray): void; }
class KS implements Sink { rays: Ray[] = []; take(r: Ray) { this.rays.push(r); } }
function main() {
    const ks = new KS();
    const s: Sink = ks;
    s.take({ start: new V(1), dir: new V(2) });
    print(ks.rays[0].start.x, ks.rays[0].dir.x);
}
