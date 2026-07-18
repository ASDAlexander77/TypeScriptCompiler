// further coverage around manual .next() driving, extending 00generator_manual_next.ts:
// interleaving two independent generator instances by hand, manually driving a generator
// from inside a plain function (not for...of), and a generator body that closes over and
// mutates outer state across suspensions.

function* gen(start: number, count: number) {
    for (let i = 0; i < count; i++) {
        yield start + i;
    }
}

// two independent instances, manually interleaved -- each must keep its own state.
function main1() {
    const a = gen(0, 3);
    const b = gen(100, 3);

    let ra = a.next();
    let rb = b.next();
    assert(ra.value == 0);
    assert(rb.value == 100);

    rb = b.next();
    ra = a.next();
    assert(rb.value == 101);
    assert(ra.value == 1);

    ra = a.next();
    rb = b.next();
    assert(ra.value == 2);
    assert(rb.value == 102);

    ra = a.next();
    rb = b.next();
    assert(ra.done);
    assert(rb.done);
}

// manually drain a generator partially from inside a plain (non-generator) helper
// function that receives it as a parameter, then continue driving the SAME iterator
// from the caller. The generator wrapper is a reference type (heap-boxed ObjectType),
// so the callee's .next() calls advance the caller's binding too -- regression
// coverage for the former value-semantics copy bug at the function-parameter boundary.
function drainTwo(it: ReturnType<typeof gen>) {
    const first = it.next();
    const second = it.next();
    return [first.value, second.value];
}

function main2() {
    const it = gen(5, 4);

    const [v0, v1] = drainTwo(it);
    assert(v0 == 5);
    assert(v1 == 6);

    // the caller's binding must observe the callee's two next() calls
    let r = it.next();
    assert(r.value == 7);
    assert(!r.done);

    r = it.next();
    assert(r.value == 8);

    r = it.next();
    assert(r.done);

    // plain assignment aliases the same generator state
    const a = gen(0, 5);
    const b = a;
    a.next(); // consumes 0
    const rb = b.next();
    assert(rb.value == 1);

    // closure capture aliases the same generator state
    const it2 = gen(100, 3);
    const drainOne = () => { it2.next(); };
    drainOne();
    const r2 = it2.next();
    assert(r2.value == 101);
}

// generator closing over outer mutable state; manual .next() calls interleaved with
// mutation of that outer state between resumptions.
let outer = 0;

function* counterFromOuter() {
    while (outer < 5) {
        yield outer;
        outer++;
    }
}

function main3() {
    outer = 0;
    const it = counterFromOuter();

    let r = it.next();
    assert(r.value == 0);

    outer = 3; // mutate captured state between manual resumptions;
               // resuming re-enters the loop body right after the yield, so
               // outer++ (-> 4) runs before the while condition is rechecked

    r = it.next();
    assert(r.value == 4);

    r = it.next();
    assert(r.done);
}

// calling .next() again after the generator has already completed keeps returning done.
function* short() {
    yield 42;
}

function main4() {
    const it = short();

    let r = it.next();
    assert(!r.done);
    assert(r.value == 42);

    r = it.next();
    assert(r.done);

    // extra calls past completion must stay done, not restart or crash
    r = it.next();
    assert(r.done);

    r = it.next();
    assert(r.done);
}

function main() {
    main1();
    main2();
    main3();
    main4();

    print("done.");
}
