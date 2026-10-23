// A global `I | null` initialized with a `C | null` - a class object or null - holds an interface
// over the object, or null. The conversion was a plain cast that read the class pointer as an
// interface, under every model. Each read follows allocations of the same size, so a freed block
// is reused.
interface I {
    x: number;
    get(): number;
}

class C {
    pad: number = 7;
    constructor(public x: number) {}

    get(): number {
        return this.x * 10;
    }
}

class Junk {
    a = -1;
    b = -2;
    c = -3;
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

@noinline function maybe(flag: boolean, x: number): C | null {
    if (flag) {
        return new C(x);
    }

    return null;
}

let some: I | null = maybe(true, 3);
let none: I | null = maybe(false, 4);

function main() {
    // fields first, before any allocation or method call: the plain cast left the globals pointing
    // at whatever it made of the class pointer, and the churn after it allocated without end
    assert(some !== null && some.x == 3, "a class object's field");
    assert(none === null, "null");

    churn();
    assert(some !== null && some.x == 3 && some.get() == 30, "a class object");
    assert(none === null, "null after the churn");
    print("done.");
}
