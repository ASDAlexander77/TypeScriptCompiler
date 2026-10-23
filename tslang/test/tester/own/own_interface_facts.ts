// -mm=own: a call through an interface that only this module can see (closed: not exported, every
// implementation defined here) gets the facts its implementations agree on, as a call of a private
// class family does (spec 31). The call destroys nothing when no implementation does, so a borrow
// of the argument's field or of an element of the container the receiver came from lives across
// it; a method that keeps its argument is given it (the caller moves it in), and a result that
// borrows an argument or `this` is the caller's to read, not to release. A direct call of such a
// method keeps its facts too, and so does a `public` method of a class no other module can name.
// churn() keeps what it allocates alive, so a block freed too early is reused before it is read.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 2000; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

// `public` members of classes no other module can name
interface Pick { pick(v: V): V; }
class PA implements Pick { public pick(v: V): V { return v; } }
class PB implements Pick { public pick(v: V): V { return v; } }
function picked(a: V) {
    const ps: Pick[] = [new PA(), new PB()];
    let s: number = 0;
    for (const p of ps) { const b = p.pick(a); s += b.x; }
    return s;
}

// the plan's shape: the loop reads an element of `sc.ts` and calls through it with `r`
interface Th { area(v: V): number; }
class A implements Th { area(v: V) { return v.x; } }
class B implements Th { area(v: V) { return v.x * 2; } }
class Sc { ts: Th[] = [new A(), new B()]; }
function tot(sc: Sc, r: V) {
    let best: number = 0;
    for (let i in sc.ts) { let a = sc.ts[i].area(r); if (a > best) best = a; }
    return best + r.x;
}

// a borrow of a field held across a call through an interface whose members destroy nothing
class H { c: V = new V(5); }
interface Op { run(h: H): number; }
class OA implements Op { run(h: H): number { return h.c.x; } }
class OB implements Op { run(h: H): number { return V.plus(h.c, h.c).x; } }
function across(o: Op, h: H) {
    const c = h.c;
    const n = o.run(h);
    churn();
    return c.x + n;
}

// an object literal implementing the interface is a member like a class
function literalMember(h: H) {
    const ops: Op[] = [new OA(), { run(h: H): number { return h.c.x * 3; } }];
    let s: number = 0;
    for (const o of ops) s += across(o, h);
    return s;
}

// a method that keeps its argument: a literal, a const, a fresh value
interface Ray { start: V; dir: V; }
interface Sink { take(r: Ray): void; }
class KS implements Sink { rays: Ray[] = []; take(r: Ray) { this.rays.push(r); } }
interface SinkV { take(v: V): void; }
class KV implements SinkV { vs: V[] = []; take(v: V) { this.vs.push(v); } }

// a result that borrows its argument, an element of `this`, a field of `this`
interface Getter { get(v: V): V; }
class G implements Getter { get(v: V): V { return v; } }
interface First { first(): V; }
class F implements First { items: V[] = [new V(3)]; first(): V { return this.items[0]; } }
interface Holder { hold(): V; }
class HH implements Holder { v: V = new V(4); hold(): V { return this.v; } }
function use(h: Holder) { const x = h.hold(); return x.x; }

function main() {
    print(tot(new Sc(), new V(3)));
    assert(tot(new Sc(), new V(3)) == 9, "the interface call's facts");

    const h = new H();
    assert(across(new OA(), h) == 10, "a field borrowed across the call");
    assert(across(new OB(), h) == 15, "a field borrowed across the other member");
    assert(literalMember(h) == 30, "a literal member");

    const ks = new KS();
    const s: Sink = ks;
    s.take({ start: new V(1), dir: new V(2) });
    const r: Ray = { start: new V(3), dir: new V(4) };
    s.take(r);
    const kv = new KV();
    const sv: SinkV = kv;
    sv.take(new V(5));
    const direct = new KV();
    direct.take(new V(6));
    churn();
    assert(ks.rays[0].start.x == 1 && ks.rays[0].dir.x == 2, "a kept literal");
    assert(ks.rays[1].start.x == 3 && ks.rays[1].dir.x == 4, "a kept const");
    assert(kv.vs[0].x == 5, "a kept value");
    assert(direct.vs[0].x == 6, "a kept value, called directly");

    const g: Getter = new G();
    const a = new V(5);
    {
        const b = g.get(a);
        assert(b.x == 5, "a borrowed argument");
    }
    const gd = new G();
    {
        const b = gd.get(a);
        assert(b.x == 5, "a borrowed argument, called directly");
    }

    const f = new F();
    const i: First = f;
    {
        const x = i.first();
        assert(x.x == 3, "a borrowed element");
        const y = f.first();
        assert(y.x == 3, "a borrowed element, called directly");
    }

    const hh = new HH();
    assert(use(hh) == 4, "a borrowed field");
    assert(picked(a) == 10, "a borrowed argument through public members");
    assert(hh.hold().x == 4, "a borrowed field, called directly");

    churn();
    const c = new V(9);
    assert(a.x == 5 && f.items[0].x == 3 && hh.v.x == 4 && c.x == 9, "what the borrows read is still there");
    print(a.x, f.items[0].x, hh.v.x, c.x);

    print("done.");
}
