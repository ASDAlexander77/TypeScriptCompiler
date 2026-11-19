// obj.prop; on its own calls the getter, as in JavaScript: the getter was erased as unused, and with it
// what it does (here, count its calls or throw)
interface HasValue {
    value: int;
}

interface HasFailing {
    failing: int;
}

class Plain {
    value = 3;
}

class Counter {
    public reads = 0;

    get value(): int {
        this.reads++;
        return 7;
    }

    get failing(): int {
        throw "read";
    }

    static count = 0;

    static get next(): int {
        Counter.count++;
        return Counter.count;
    }
}

function main() {
    const c = new Counter();
    c.value;
    c.value;
    assert(c.reads == 2, "a getter read as a statement runs");
    (c.value);
    assert(c.reads == 3, "in parentheses too");

    Counter.next;
    assert(Counter.count == 1, "a static getter");

    // an interface's field, which the class gives by a getter
    const h: HasValue = c;
    h.value;
    assert(c.reads == 4, "through an interface");

    let threw = false;
    try {
        c.failing;
    } catch (e) {
        threw = true;
    }

    assert(threw, "a getter that throws, throws");

    let threwThroughInterface = false;
    const f: HasFailing = c;
    try {
        f.failing;
    } catch (e) {
        threwThroughInterface = true;
    }

    assert(threwThroughInterface, "through an interface, in a try");

    // a plain field read so is only a read
    const p: HasValue = new Plain();
    p.value;
    assert(p.value == 3, "a field");
    print("done.");
}
