// A lambda captures what its body reads, and the body is read by the discovery pass - the dummy
// run that fills passResult->outerVariables. `new C(...)` took a shortcut there: it created the
// instance without resolving the constructor, and so never walked the constructor arguments at
// all. A variable read ONLY inside those arguments was therefore never registered as captured,
// and the real pass emitted the read against the enclosing function's own value from inside the
// lambda - "'ts.Load' op using value defined outside the region", because ts.Func is
// IsolatedFromAbove. It failed to compile under every memory model, so it is not a refcounting
// bug; it is recorded in docs/reference-counting-evaluation.md section 9.54 because it silently
// shaped the benchmarks written there.
//
// Every case reads the captured thing ONLY inside the `new` arguments. Reading it anywhere else
// in the same lambda - even once - registers the capture and hides the bug entirely, which is
// what `alsoReadOutsideTheNew` below is here to say.

class Color {
    r: number;
    g: number;

    constructor(r: number, g: number) {
        this.r = r;
        this.g = g;
    }
}

class Boxed {
    v: number;

    constructor(v: number) {
        this.v = v;
    }
}

function capturedParam(a: number): number {
    const make = (k: number) => { return new Boxed(a + k); };

    return make(1.0).v;
}

function capturedField(c: Color): number {
    const make = (k: number) => { return new Boxed(c.r + k); };

    return make(1.0).v;
}

// A `let` the lambda only reads is still captured by value; the cell path is exercised by
// `capturedMutableLocal` below.
function capturedLocal(): number {
    let base = 10.0;
    const make = (k: number) => { return new Boxed(base + k); };

    return make(1.0).v;
}

// A variable the enclosing function writes after the lambda is built has to be captured by
// reference - the lambda reads the cell, not a copy - so this goes through a different capture
// shape than the three above.
function capturedMutableLocal(): number {
    let base = 10.0;
    const make = (k: number) => { return new Boxed(base + k); };
    base = 20.0;

    return make(1.0).v;
}

// More than one argument, and more than one captured variable, so a partial walk of the arguments
// would still be caught.
function severalArgumentsAndCaptures(x: number, y: number): number {
    const make = () => { return new Color(x, y); };
    const c = make();

    return c.r * 10.0 + c.g;
}

// The argument of the outer `new` is itself a `new` reading a captured variable, so the walk has
// to descend rather than just visit each argument's top node.
function nestedNew(a: number): number {
    const make = () => { return new Boxed(new Boxed(a).v + 1.0); };

    return make().v;
}

// A lambda inside a lambda: the inner one reads a variable that belongs to the outermost
// function, which reaches it through the middle lambda's own capture.
function nestedLambdas(a: number): number {
    const outer = () => {
        const inner = () => { return new Boxed(a + 1.0); };
        return inner().v;
    };

    return outer();
}

class Owner {
    scale: number;

    constructor(scale: number) {
        this.scale = scale;
    }

    // `this` captured by a lambda in a method, read only inside the `new` arguments.
    build(k: number): number {
        const make = () => { return new Boxed(this.scale * k); };

        return make().v;
    }
}

// The control: the same capture read outside the `new` as well. This compiled all along, which is
// why the bug survived - it only bites when the arguments are the variable's only appearance.
function alsoReadOutsideTheNew(a: number): number {
    const make = (k: number) => {
        let seen = a;
        return new Boxed(a + k + seen * 0.0);
    };

    return make(1.0).v;
}

function main() {
    assert(capturedParam(2.0) == 3.0, "a captured parameter read only in `new` arguments");
    assert(capturedField(new Color(2.0, 0.0)) == 3.0, "a captured object's field in `new` arguments");
    assert(capturedLocal() == 11.0, "a captured local in `new` arguments");
    assert(capturedMutableLocal() == 21.0, "a captured mutable local in `new` arguments");
    assert(severalArgumentsAndCaptures(3.0, 4.0) == 34.0, "several captures across several arguments");
    assert(nestedNew(2.0) == 3.0, "a `new` inside another `new`'s arguments");
    assert(nestedLambdas(2.0) == 3.0, "a lambda inside a lambda reaching the outer function");
    assert(new Owner(3.0).build(2.0) == 6.0, "`this` captured by a lambda in a method");
    assert(alsoReadOutsideTheNew(2.0) == 3.0, "the same capture read outside the `new` too");

    print("done.");
}
