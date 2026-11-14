// A closure stored into a field of the object it captures: `c.onTick = () => { c.seen++ }`. Under
// -mm=own the closure's box owned `c`, which owned the closure, and clearing the field freed `c`
// itself. The box borrows the object, as one that captures `this` in a method does: the field owns
// the closure and the object bounds it.

class Counter {
    onTick: ((n: number) => void) | null = null;
    onDone: (n: number) => void = (n: number) => {};
    count = 0;
    seen = 0;

    tick() {
        this.count++;
        if (this.onTick !== null) {
            this.onTick(this.count);
        }

        this.onDone(this.count);
    }
}

function testNullableField() {
    const c = new Counter();
    c.onTick = (n: number) => {
        c.seen += n;
    };
    c.tick();
    assert(c.seen == 1);
    c.onTick = null;
    c.tick();
    assert(c.seen == 1);
    assert(c.count == 2);
}

function testPlainField() {
    const c = new Counter();
    c.onDone = (n: number) => {
        c.seen += n;
    };
    c.tick();
    c.tick();
    assert(c.seen == 3);
    c.onDone = (n: number) => {};
    c.tick();
    assert(c.seen == 3);
    assert(c.count == 3);
}

function testReplaced() {
    const c = new Counter();
    for (let i = 0; i < 100; i++) {
        c.onTick = (n: number) => {
            c.seen += n;
        };
        c.tick();
    }

    assert(c.count == 100);
    assert(c.seen == 5050);
}

function main() {
    testNullableField();
    testPlainField();
    testReplaced();
    print("done.");
}
