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

// manually drain a generator entirely from inside a plain (non-generator) helper
// function that receives it as a parameter.
//
// NOTE: driving the SAME iterator further from the caller after it has been passed
// into and mutated by a helper function is a known, separate bug (not covered here):
// generator objects have value semantics and are copied across a function-parameter
// boundary, so .next() calls made inside the callee do not advance the caller's
// binding. That is unlike a same-function const local (see main1/00generator_manual_next.ts),
// which works via an alloca-caching mechanism scoped to a single function body. Fixing
// this would require pass-by-reference semantics for generator-typed parameters at the
// ABI level in mlirGenFunctionParams -- out of scope for this regression file.
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
