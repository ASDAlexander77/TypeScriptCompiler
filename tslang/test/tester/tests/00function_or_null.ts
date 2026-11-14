// `(() => T) | null`: a function or null. It has no tag - it is the function's own value, null when
// it has no code to run. A function could not be stored in one ("invalid cast from () => number to
// () => number | null"), `if (f)` did not narrow it and `f()` was "not a function to call".

class Holder {
    f: (() => number) | null = null;
    g: (() => number) | undefined = undefined;
    h?: () => number;

    set(t: () => number) {
        this.f = t;
    }

    run(): number {
        if (this.f) {
            return this.f();
        }

        return -1;
    }
}

class Sink {
    seen = 0;
}

let seenTotal = 0;

class Counter {
    onTick: ((n: number) => void) | null = null;
    count = 0;

    tick() {
        this.count++;
        if (this.onTick !== null) {
            this.onTick(this.count);
        }
    }
}

function run(f: (() => string) | null): string {
    if (!f) {
        return "none";
    }

    return f();
}

function testField() {
    const h = new Holder();
    assert(h.run() == -1);
    h.set((): number => 42);
    assert(h.run() == 42);

    h.f = (): number => 1;
    h.g = (): number => 2;
    h.h = (): number => 3;
    if (h.f) assert(h.f() == 1);
    if (h.g) assert(h.g() == 2);
    if (h.h) assert(h.h() == 3);

    h.f = null;
    assert(h.run() == -1);
}

function testLocal() {
    let f: (() => number) | null = null;
    assert(!f);
    assert(f === null);
    const t = (): number => 42;
    f = t;
    assert(f !== null);
    if (f) {
        assert(f() == 42);
    } else {
        assert(false);
    }

    f = null;
    assert((f ? "set" : "null") == "null");
}

function testCompare() {
    // the closure keeps another object: one that captured the counter it is stored in would own
    // its own owner
    const c = new Counter();
    const sink = new Sink();
    c.tick();
    c.onTick = (n: number) => {
        sink.seen += n;
        seenTotal += sink.seen;
    };
    c.tick();
    c.tick();
    assert(seenTotal == 7);
    c.onTick = null;
    c.tick();
    assert(seenTotal == 7);
    assert(c.count == 4);
}

// null adds no field: the union is as wide as the function value, which holds its code and what it
// runs with (a closure's captures, a method's `this`)
function testLayout() {
    const plain = (): number => 1;
    let fn: () => number = plain;
    let fnOrNull: (() => number) | null = plain;
    assert(sizeof(fnOrNull) == sizeof(fn));
    assert(fn() == 1);

    const first = "first";
    const second = "second";
    let capturing: () => string = () => first;
    let capturingOrNull: (() => string) | null = () => second;
    assert(sizeof(capturingOrNull) == sizeof(capturing));
    assert(capturing() == "first");
    if (capturingOrNull) {
        assert(capturingOrNull() == "second");
    } else {
        assert(false);
    }

    capturingOrNull = null;
    assert(capturingOrNull === null);
}

function testParameter() {
    const name = "captured";
    assert(run(null) == "none");
    assert(run(() => name) == "captured");
}

function main() {
    testField();
    testLocal();
    testCompare();
    testLayout();
    testParameter();
    print("done.");
}
