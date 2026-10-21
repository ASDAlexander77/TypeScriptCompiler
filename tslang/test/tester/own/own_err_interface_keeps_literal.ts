// -mm=own rejects: a method that keeps its argument, reached through an interface. The call
// through the interface is not given the method's facts, so the caller would release the literal
// the method keeps (a double free). Until facts reach interface calls (G9), the method's body is
// the error.
class V {
    constructor(public x: number) {}
}
interface Ray { start: V; dir: V; }
interface Sink { take(r: Ray): void; }
class KS implements Sink { rays: Ray[] = []; take(r: Ray) { this.rays.push(r); } }
function main() {
    const ks = new KS();
    const s: Sink = ks;
    s.take({ start: new V(1), dir: new V(2) });
    print(ks.rays[0].start.x, ks.rays[0].dir.x);
}
