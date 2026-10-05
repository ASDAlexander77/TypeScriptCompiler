// -mm=rc: a parameter the body assigns owns what it is assigned (#512). The slot of a parameter is borrowed
// from the caller and took no reference, so `x = new B(1)` stored an instance the block that made it then
// released: read after that block - the next loop condition, a statement after an `if` - it was freed memory.
// `while (x instanceof A) { x = new B(1); }` crashed; the rest read garbage once the heap was reused.
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

function captured(x: B) {
    const g = () => x.b;
    {
        x = new B(9);
    }

    churn();
    return g();
}

function inClosure(x: B) {
    const set = () => {
        x = new B(8);
    };
    set();
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

function nullish(x: B | undefined) {
    {
        x ??= new B(13);
    }

    churn();
    return x.b;
}

function strings(s: string) {
    for (let i = 0; i < 3; i++) {
        s = s + i;
    }

    churn();
    return s;
}

function* generator(x: B) {
    {
        x = new B(1);
    }

    churn();
    yield x.b;
    {
        x = new B(2);
    }

    churn();
    yield x.b;
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
    assert(captured(new B(0)) == 9, "captured, assigned in a block");
    assert(inClosure(new B(0)) == 8, "assigned in a closure");
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
    assert(nullish(undefined) == 13, "??=");
    assert(strings("s") == "s012", "string");
    assert(unassigned(new B(14)) == 14, "unassigned");

    let yielded = 0;
    for (const v of generator(new B(0))) {
        yielded = yielded * 10 + v;
    }

    assert(yielded == 12, "generator");
    assert((await asynchronous(new B(0))) == 3, "async function");
    assert(rest(new B(0)) == 4, "rest parameter");

    print("done.");
}
