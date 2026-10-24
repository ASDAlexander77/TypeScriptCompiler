// -mm=own: a tuple that holds borrows and owns nothing is a record with no box (spec 36), and so is a
// literal given only to calls as a temporary: `len({ start: new Vec(1, 2), dir: new Vec(3, 4) })`
// borrows the two new vectors, which keep their own release after the call. A `@value` record held
// in a `const` that borrows a `let` twice is read through the local in place. A closure in a `@value` constructor that reads
// `this`, whose fields are numbers, needs no owner. churn() inside the callee and before each read
// reuses the blocks a vector freed too early would leave.
class Vec {
    constructor(public x: number, public y: number) {}
}

@value
interface Ray {
    readonly start: Vec;
    readonly dir: Vec;
}

type Pair = { a: Vec; b: Vec };

// a value that owns its vector (a fresh one moved in), whose number field is printed straight:
// converting the number to a string takes nothing of the value
@value
interface Owned {
    readonly v: Vec;
    readonly n: number;
}

function owned(): Owned {
    return { v: new Vec(3, 4), n: 2 };
}

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

function churn() {
    const keep: Vec[] = [];
    for (let i = 0; i < 64; i++) {
        keep.push(new Vec(77777, 77777));
    }

    return keep.length;
}

function len(r: Ray) {
    churn();
    return r.start.x + r.dir.y;
}

function sum(p: Pair) {
    churn();
    return p.a.x + p.b.y;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        total += len({ start: new Vec(i, 0), dir: new Vec(0, 1) });
        total += sum({ a: new Vec(1, 0), b: new Vec(0, i) });
    }

    print(total);
    assert(total == 2 * 499500 + 2000);

    let v = new Vec(5, 6);
    const r: Ray = { start: v, dir: v };
    churn();
    assert(r.start.x == 5 && r.dir.y == 6);

    assert(new W(4).b == 5);

    const o = owned();
    print(o.n);
    churn();
    assert(o.v.x == 3 && o.n == 2);
    print("done.");
}
