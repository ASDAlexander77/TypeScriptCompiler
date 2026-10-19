// regression test: an object literal's field holding a function that does not take
// the object as its `this` - a capturing closure (bound function), a hybrid function
// (a type-literal annotated field) or a plain global function - fills an interface
// member declared as a function-typed field (`fn: () => int`) or as a method
// (`fn(): int`). The interface calls its slot with the object as `this`, so the
// closure's captured context used to be dropped (garbage, or a crash) and a global
// function received the object as its first argument. The slot now calls a thunk
// that reads the field from the object and calls it.

interface HasFn {
    fn: () => int;
}

interface HasMethod {
    fn(): int;
}

interface Adder {
    add: (a: int) => int;
}

interface Counted {
    n: number;
    fn: () => int;
}

interface MaybeFn {
    fn?: () => int;
}

interface Pair {
    first: (a: int) => int;
    second(a: int, b: int): int;
}

function plusOne(a: int): int {
    return a + 1;
}

function times(a: int, b: int): int {
    return a * b;
}

// the literal is built for the interface: the closure keeps its own type in it
function make(n: int): HasFn {
    const cl = () => n * 3;
    return { fn: cl };
}

function main() {
    let k = 5;
    const cl = () => k + 1;

    // a capturing closure filling a function-typed field
    const o = { fn: cl };
    const h: HasFn = o;
    print(h.fn());
    assert(h.fn() == 6);

    // the closure reads the captured variable, not a copy of it
    k = 10;
    assert(h.fn() == 11);

    // ... and filling a method
    const m: HasMethod = o;
    assert(m.fn() == 11);

    // a closure taking a parameter
    let base = 100;
    const o2 = { add: (a: int) => a + base };
    const add = o2.add;
    const o3 = { add: add };
    const a1: Adder = o3;
    assert(a1.add(1) == 101);

    // a type-literal annotated field holds a hybrid function
    const o4: { add: (a: int) => int } = { add: (a: int) => a + k };
    const a2: Adder = o4;
    assert(a2.add(1) == 11);

    // a plain function taking parameters gets them, not the object
    const o5 = { add: plusOne };
    const a3: Adder = o5;
    assert(a3.add(10) == 11);

    const o6 = { first: plusOne, second: times };
    const p: Pair = o6;
    assert(p.first(1) == 2);
    assert(p.second(3, 4) == 12);

    // an s32 field read as `number` forces a clone of the object: the closure survives it
    const o7 = { n: 7, fn: cl };
    const c: Counted = o7;
    assert(c.n == 7);
    assert(c.fn() == 11);

    // a literal typed by the interface it is assigned to
    const h9: HasFn = { fn: cl };
    assert(h9.fn() == 11);

    // the closure outlives the function that made it
    const made = make(4);
    assert(made.fn() == 12);
    const arr: HasFn[] = [make(1), make(2)];
    let sum = 0;
    for (const x of arr) sum += x.fn();
    assert(sum == 9);

    // an optional function-typed field
    const mf: MaybeFn = o;
    assert(mf.fn?.() == 11);

    // an object literal method still sees its object
    const o8 = { v: 3, fn() { return this.v * 2; } };
    const h8: HasFn = o8;
    assert(h8.fn() == 6);

    print("done.");
}
