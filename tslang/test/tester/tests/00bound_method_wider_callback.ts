// a bound method given as a function type with more parameters than the method takes, the extra ones first

class Counter {
    n = 0;
    bump() {
        this.n++;
    }
    add(x: number): number {
        this.n += x;
        return this.n;
    }
}

function callWithObject(f: (o: object) => void, o: object) {
    f(o);
}

function callWithObjectAndNumber(f: (o: object, x: number) => number, o: object, x: number) {
    return f(o, x);
}

function main() {
    const c = new Counter();
    const other = new Counter();
    for (let i = 0; i < 1000; i++) {
        callWithObject(c.bump, other);
    }

    assert(c.n == 1000, "the method's own this, not the argument");
    assert(other.n == 0);

    assert(callWithObjectAndNumber(c.add, other, 5) == 1005, "the last argument goes to the method");

    const user = {
        firstName: "World",
        count: 0,
        sayHi() {
            this.count++;
        },
    };

    for (let i = 0; i < 100; i++) {
        callWithObject(user.sayHi, user);
    }

    assert(user.count == 100);

    print("done.");
}
