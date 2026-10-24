// A @value value holding references: copying it copies them as a tuple copy does (rc retains each
// one), through locals, an array, a generic and a closure, and a constructor's stores into its fields
// count as stores into an owned local's. A closure captures the binding of a @value local, not a
// snapshot; one in a @value constructor reads `this` as the value set so far (and is only called
// there), and one in a method takes a copy of `this`. Each read of a copy follows `churn()`, which
// reuses the blocks a missing retain would have freed (same-size objects), and reads the field first.
class Node {
    constructor(public n: int) {}
}

@value
interface Rec {
    readonly node: Node;
    readonly w: number;
}

@value
class V {
    constructor(public readonly x: number) {}

    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

// a field computed by a closure that reads `this` in the constructor
@value
class W {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        const f = () => this.a + 1;
        this.b = f();
    }
}

// ...and through a closure inside a closure, several values in one frame
@value
class W2 {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        const f = () => {
            const h = () => this.a + 1;
            return h();
        };
        this.b = f();
    }
}

// a constructor storing a new reference in a field, one storing its parameter (a parameter property
// and an assignment)
@value
class Holder {
    readonly node: Node;

    constructor(n: int) {
        this.node = new Node(n);
    }
}

@value
class HolderP {
    constructor(public readonly node: Node) {}
}

@value
class HolderA {
    readonly node: Node;

    constructor(n: Node) {
        this.node = n;
    }
}

// closures in methods, which take a copy of `this`: one returned, one inside another
@value
class Box {
    constructor(public readonly node: Node) {}

    reader() {
        return () => this.node.n;
    }

    nested() {
        const f = () => {
            const h = () => this.node.n;
            return h();
        };
        return f();
    }
}

// a @value interface of references, held in an ordinary class's field
@value
interface Ray {
    readonly start: Node;
    readonly dir: Node;
}

class Hit {
    r: Ray;

    constructor(r: Ray) {
        this.r = r;
    }
}

function churn() {
    let s = 0;
    for (let i = 0; i < 64; i++) {
        const t = new Node(i);
        s += t.n;
    }
    return s;
}

function id<T>(t: T) {
    return t;
}

function makeReader(r: Rec) {
    return () => r.node.n;
}

function makeHolder(i: int) {
    return new Holder(i);
}

function makeHolderP(i: int) {
    return new HolderP(new Node(i));
}

function makeHolderA(i: int) {
    return new HolderA(new Node(i));
}

// a value's fields hold what its constructor stored in them, a new reference too
function constructed() {
    const h = new Holder(7);
    churn();
    assert(h.node.n == 7);

    let sum = 0;
    for (let i = 0; i < 100; i++) {
        const a = makeHolder(i);
        const b = makeHolderP(i);
        const c = makeHolderA(i);
        churn();
        sum += a.node.n + b.node.n + c.node.n;
    }

    assert(sum == 3 * 4950);
}

// a closure a method returned keeps its copy of `this` after the value it was called on is overwritten
function methodClosures() {
    const boxes: Box[] = [new Box(new Node(9))];
    const r = boxes[0].reader();
    boxes[0] = new Box(new Node(-1));
    churn();
    assert(r() == 9);
    assert(new Box(new Node(3)).nested() == 3);
}

// several values in one frame, each from a constructor whose closures read `this`
function closuresInConstructors(k: number) {
    const p = new W2(k);
    const q = new W2(k + 10);
    const r = new W2(k + 20);
    assert(p.b == k + 1);
    assert(q.b == k + 11);
    assert(r.b == k + 21);
    assert(new W(4).b == 5);
}

// copies of an array's values outlive the array's reference to their Nodes
function copies() {
    const recs: Rec[] = [];
    for (let i = 0; i < 1000; i++) {
        recs.push({ node: new Node(i), w: i });
    }

    let sum = 0;
    for (let i = 0; i < 1000; i++) {
        const a = recs[i];
        const b = recs[i];
        recs[i] = { node: new Node(-1), w: 0 };
        churn();
        sum += a.node.n + b.node.n;
    }

    assert(sum == 999000);
}

// ...and so does one passed through a generic
function throughGeneric() {
    const recs: Rec[] = [{ node: new Node(11), w: 1 }];
    const c = id(recs[0]);
    recs[0] = { node: new Node(-1), w: 0 };
    churn();
    assert(c.node.n == 11);
}

// a literal of fields read from a value in a class's field, and a const of one, keep them after the
// field changes
function rayFromHit() {
    const h = new Hit({ start: new Node(3), dir: new Node(4) });
    const o: Ray = { start: h.r.start, dir: h.r.dir };
    const s = h.r.start;
    h.r = { start: new Node(5), dir: new Node(6) };
    churn();
    assert(o.start.n == 3);
    assert(o.dir.n == 4);
    assert(s.n == 3);
}

// a closure that captures a Rec reads its own Node after the original's was overwritten
function closures() {
    const recs: Rec[] = [{ node: new Node(7), w: 1 }];
    const r = recs[0];
    const f = () => r.node.n;
    const g = makeReader(recs[0]);
    recs[0] = { node: new Node(-1), w: 0 };
    churn();
    assert(f() == 7);
    assert(g() == 7);
}

function main() {
    copies();
    throughGeneric();
    rayFromHit();
    closures();
    constructed();
    methodClosures();
    closuresInConstructors(1);

    assert([new V(1), new V(2), new V(3)].reduce((acc, x) => V.plus(acc, x), new V(0)).x == 6);

    let v = new V(1);
    const f = () => v.x;
    v = new V(2);
    assert(f() == 2);

    print("done.");
}
