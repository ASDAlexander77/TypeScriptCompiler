// Regression coverage for docs/object-literal-boxing-design.md PR B: object literals
// with a method/accessor are now boxed as a reference-typed ObjectType (same recipe
// as the generator wrapper, PR #245), so every alias of such a literal shares state.
// Pure-data literals (no methods) deliberately keep value semantics -- not tested here.

// plain assignment aliases the same object.
function main1() {
    const a = { x: 1, inc() { this.x = this.x + 1; } };
    const b = a;
    b.inc();
    assert(a.x == 2);
}

// passing to a function parameter aliases the caller's object.
function bump(o: { x: number; inc: () => void }) {
    o.inc();
}

function main2() {
    const a = { x: 10, inc() { this.x = this.x + 1; } };
    bump(a);
    assert(a.x == 11);
}

// closure capture aliases the captured object.
function main3() {
    const a = { x: 100, inc() { this.x = this.x + 1; } };
    const bumpA = () => { a.inc(); };
    bumpA();
    assert(a.x == 101);
}

// array element aliasing: reading the same index twice yields the same identity.
function main4() {
    const arr = [{ x: 1, inc() { this.x = this.x + 1; } }];
    arr[0].inc();
    assert(arr[0].x == 2);
}

// object nested inside another object aliases correctly.
function main5() {
    const outer = {
        inner: { x: 1, inc() { this.x = this.x + 1; } },
    };
    const innerAlias = outer.inner;
    innerAlias.inc();
    assert(outer.inner.x == 2);
}

// accessor (get/set) mutating `this` is visible through an alias too.
// NOTE: backing field uses a float literal (1.0), not an integer literal (1),
// to avoid a pre-existing, unrelated bug: an accessor whose declared parameter
// type (number) differs from the backing field's inferred type (si32 for an
// integer literal) passes the setter's argument un-cast, corrupting the write.
// Reproduces on unmodified main (pre-dates this change); not fixed here.
function main6() {
    const a = {
        _x: 1.0,
        get x() { return this._x; },
        set x(v: number) { this._x = v; },
    };
    const b = a;
    b.x = 42;
    assert(a.x == 42);
}

// global const object literal with a mutating method (generalizes the #246 repro
// in 00global_const_object_method.ts to the general boxing flip, not just the
// special-cased BoxAsObject path).
const globalCounter = { count: 0, inc() { this.count = this.count + 1; } };

function main7() {
    globalCounter.inc();
    globalCounter.inc();
    assert(globalCounter.count == 2);
}

// conditional-expression merge of two same-shape (method-bearing) literals: both
// branches must produce the same structural ObjectType so the ternary type-checks
// and the resulting binding still has working method calls either way.
function pick(useFirst: boolean) {
    return useFirst
        ? { x: 1, inc() { this.x = this.x + 1; } }
        : { x: 2, inc() { this.x = this.x + 1; } };
}

function main8() {
    const a = pick(true);
    a.inc();
    assert(a.x == 2);

    const b = pick(false);
    b.inc();
    assert(b.x == 3);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    main5();
    main6();
    main7();
    main8();

    print("done.");
}
