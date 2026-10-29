// A field that refers back to its own object (spec 70): an arrow reading `this`, a bound method of
// `this`. Its closure's box borrows the object - the field owns the closure, the box frees itself
// alone - and the object bounds every use of it. Stored by a field initializer, a constructor and a
// method, in a class and one extending it; read and called through the object, after it is moved.
// Each step churns, so anything freed too early reads garbage; every model runs it.
class Junk {
    constructor(public n: number) {}
}

function churn() {
    const junk: Junk[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Junk(999));
}

class Counter {
    n: number = 1;
    get = () => this.n;
    add = (k: number) => this.n + k;
    twice = () => this.get() + this.get();
    scaled = (k: number) => this.get() * k + this.value();

    value() {
        return this.n * 10;
    }

    bound = this.value;

    double() {
        this.get = () => this.n * 2;
    }
}

class Wider extends Counter {
    more = () => this.add(100);

    constructor() {
        super();
        this.twice = () => this.n + 1000;
    }
}

function make(n: number) {
    const c = new Counter();
    c.n = n;
    return c;
}

function main() {
    const c = new Counter();
    churn();
    assert(c.get() == 1 && c.add(2) == 3 && c.twice() == 2, "arrow fields");
    assert(c.scaled(3) == 13, "an arrow calling another and a method");
    assert(c.bound() == 10, "a bound method field");

    c.n = 5;
    churn();
    assert(c.get() == 5 && c.bound() == 50, "they read the object when called");

    c.double();
    churn();
    assert(c.get() == 10, "assigned in a method");

    c.get = () => 7;
    assert(c.get() == 7, "assigned from outside");

    const w = new Wider();
    churn();
    assert(w.more() == 101 && w.twice() == 1001, "a derived class's, and one its constructor stores");

    let moved = make(3);
    churn();
    let other = moved;
    assert(other.get() == 3 && other.add(1) == 4, "after the object moved");

    const all: Counter[] = [];
    for (let i = 0; i < 100; i++) {
        all.push(make(i));
    }

    churn();
    let total = 0;
    for (const item of all) {
        total += item.get();
    }

    assert(total == 4950, "objects an array holds");

    print("done.");
}
