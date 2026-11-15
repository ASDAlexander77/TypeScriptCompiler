// -mm=own: a parameter the body assigns is one the function keeps. It starts a local of its own, which owns what
// it holds - the caller's value first, then each value assigned - so the callers give the argument up, as they do
// to a function that stores it in a field, and may not read it after the call (own_err_param_assigned). The
// same shapes as tests/00param_assigned_owned.ts, less the ones own cannot follow yet: a parameter a closure
// captures or assigns, one a generator assigns, and one given a value by `??=`.
class A {
    constructor(public a: number) {}
}

class B {
    constructor(public b: number) {}
}

// reuses freed blocks, so a read of one sees another value rather than the old one
function churn() {
    let fill: B[] = [];
    for (let i = 0; i < 200; i++) {
        fill.push(new B(1000 + i));
    }

    return fill.length;
}

function loopInstanceOf(x: A | B) {
    let n = 0;
    while (x instanceof A) {
        x = new B(1);
        n++;
    }

    return n;
}

function forInstanceOf(x: A | B) {
    let n = 0;
    for (; x instanceof A; x = new B(1)) {
        n++;
    }

    return n;
}

function inIf(x: B, c: boolean) {
    if (c) {
        x = new B(7);
    }

    churn();
    return x.b;
}

function inLoop(x: B) {
    for (let i = 0; i < 3; i++) {
        x = new B(i);
    }

    churn();
    return x.b;
}

function optional(x?: B) {
    if (!x) {
        x = new B(7);
    }

    churn();
    return x.b;
}

function withDefault(x: B = new B(1)) {
    {
        x = new B(6);
    }

    churn();
    return x.b;
}

function returned(x: B) {
    {
        x = new B(5);
    }

    return x;
}

function destructured(x: B) {
    {
        [x] = [new B(4)];
    }

    churn();
    return x.b;
}

function forOf(x: B) {
    const arr = [new B(1), new B(3)];
    for (x of arr) {
    }

    churn();
    return x.b;
}

class M {
    m(x: B) {
        for (let i = 0; i < 2; i++) {
            x = new B(10 + i);
        }

        churn();
        return x.b;
    }
}

const arrow = (x: B) => {
    {
        x = new B(12);
    }

    churn();
    return x.b;
};

function strings(s: string) {
    for (let i = 0; i < 3; i++) {
        s = s + i;
    }

    churn();
    return s;
}

async function asynchronous(x: B) {
    {
        x = new B(3);
    }

    await 0;
    churn();
    return x.b;
}

function rest(...xs: B[]) {
    {
        xs = [new B(4)];
    }

    churn();
    return xs[0].b;
}

// given a local's value: the parameter's local takes it over, and the local is free to hold another
function fromLocal(x: B) {
    let n = new B(30);
    x = n;
    n = new B(31);
    churn();
    return x.b + n.b;
}

function unassigned(x: B) {
    churn();
    return x.b;
}

async function main() {
    assert(loopInstanceOf(new A(5)) == 1, "while on instanceof");
    assert(loopInstanceOf(new B(5)) == 0, "while on instanceof, not entered");
    assert(forInstanceOf(new A(5)) == 1, "for on instanceof");
    assert(inIf(new B(0), true) == 7, "assigned in an if");
    assert(inIf(new B(3), false) == 3, "not assigned");
    assert(inLoop(new B(0)) == 2, "assigned in a loop");
    assert(optional() == 7, "optional, assigned");
    assert(optional(new B(2)) == 2, "optional, given");
    assert(withDefault() == 6, "default, assigned");
    assert(withDefault(new B(0)) == 6, "default given, assigned");

    const r = returned(new B(0));
    churn();
    assert(r.b == 5, "returned after assignment");

    assert(destructured(new B(0)) == 4, "destructuring assignment");
    assert(forOf(new B(0)) == 3, "for...of into the parameter");
    assert(new M().m(new B(0)) == 11, "method");
    assert(arrow(new B(0)) == 12, "arrow function");
    assert(strings("s") == "s012", "string");
    assert(fromLocal(new B(0)) == 61, "given a local's value");
    assert(unassigned(new B(14)) == 14, "unassigned");

    assert((await asynchronous(new B(0))) == 3, "async function");
    assert(rest(new B(0)) == 4, "rest parameter");

    // an argument the caller holds in a local moves in, and the local may be given another value after
    let held = new B(20);
    assert(inLoop(held) == 2, "a local moved in");
    held = new B(21);
    assert(unassigned(held) == 21, "and assigned again");
    assert(held.b == 21, "a parameter nothing assigns is only lent");

    print("done.");
}
