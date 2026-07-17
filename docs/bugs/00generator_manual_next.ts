// regression test: calling .next() manually on a real `function*` generator (as opposed
// to driving it via `for...of`) used to never advance the generator's internal state.
//
// root cause: a `const` binding was stored in the symbol table as a bare SSA value with
// no backing stack storage. Each `g.next()` property access needed a ref to recover
// `this` for the bound ".next" method, and without real storage it fell back to
// allocating a fresh temporary copy of `g` -- re-seeded from the pristine, never-mutated
// original -- on every single call site. So every manual `.next()` call restarted the
// generator instead of resuming it. `for...of` happened to work because its lowering
// materializes the generator object into one persistent local up front and reuses it.
//
// fix: a const whose value is a tuple with a bound-method field (e.g. a generator or
// closure object) now gets real stack storage, matching what for...of already relied on.

function* gen() {
    for (let i = 0; i < 5; i++) {
        yield i;
    }
}

function main() {
    const g = gen();

    let r = g.next();
    assert(!r.done);
    assert(r.value == 0);

    r = g.next();
    assert(!r.done);
    assert(r.value == 1);

    r = g.next();
    assert(!r.done);
    assert(r.value == 2);

    r = g.next();
    assert(!r.done);
    assert(r.value == 3);

    r = g.next();
    assert(!r.done);
    assert(r.value == 4);

    r = g.next();
    assert(r.done);

    print("done.");
}
