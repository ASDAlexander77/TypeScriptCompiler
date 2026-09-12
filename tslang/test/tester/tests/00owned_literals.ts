// An array literal's data block, and a boxed object literal's storage, release what they hold
// when they die - but neither is filled through an assignment, so before this neither took a
// reference to what it captured. The block gave up references it never had.
//
// That is not a leak. It is an over-release, and unlike everything else in step 5 it was
// reachable today, with the birth-reference slack still in place. An element seeded by a literal
// sits one below an equivalent field: the first overwrite's release is masked by the unconsumed
// birth reference, and a second release of the same value takes it past zero and frees it while
// a local still holds it. Reduced, it printed 7, 7, 7, 0 - the last read landing on freed memory.
//
// So unlike 00owned_fields.ts, and like 00owned_elements.ts, these have teeth right now. Checked
// case by case against the compiler as it stood before this change: every case below returned a
// wrong answer - 3 where 10 was due, 1 where 7 was - except plainLiteral, which is the control
// and is meant to pass either way. What a case needs to bite is one value reaching two slots that
// are both later overwritten - two literals sharing it, or one literal holding it twice. A single
// overwrite is not enough; the slack still covers that one.
//
// See docs/reference-counting-evaluation.md section 9.21.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

// the reduced case, exactly as it was found
function twoArraysShareAValue() {
    let kept = new Leaf(7);
    let a = [kept];
    let b = [kept];
    a[0] = new Leaf(1);
    b[0] = new Leaf(2);

    return kept.n + a[0].n + b[0].n;
}

// a third holder, to show it is not specific to the count two
function threeArraysShareAValue() {
    let kept = new Leaf(7);
    let a = [kept];
    let b = [kept];
    let c = [kept];
    a[0] = new Leaf(1);
    b[0] = new Leaf(1);
    c[0] = new Leaf(1);

    return kept.n;
}

// one literal holding the same value in two of its slots
function oneArrayHoldsItTwice() {
    let x = new Leaf(1);
    let arr = [x, x];
    arr[0] = new Leaf(3);
    arr[1] = new Leaf(4);

    return x.n + arr[0].n + arr[1].n;
}

// the object-literal half: a literal with a method is boxed as a reference type, and its storage
// releases its fields the same way an array's data block releases its elements
function twoObjectLiteralsShareAValue() {
    let kept = new Leaf(7);
    let a = { item: kept, touch() { return this.item.n; } };
    let b = { item: kept, touch() { return this.item.n; } };
    a.item = new Leaf(1);
    b.item = new Leaf(2);

    return kept.n + a.item.n + b.item.n;
}

// one boxed literal capturing the same value in two of its fields
function oneObjectHoldsItTwice() {
    let x = new Leaf(5);
    let o = { p: x, q: x, touch() { return this.p.n; } };
    o.p = new Leaf(1);
    o.q = new Leaf(2);

    return x.n + o.p.n + o.q.n;
}

// a literal whose elements are themselves literals, so the retain runs on an array value rather
// than a class reference
function nestedLiterals() {
    let leaf = new Leaf(9);
    let outer = [[leaf], [leaf]];
    outer[0] = [new Leaf(1)];
    outer[1] = [new Leaf(2)];

    return leaf.n + outer[0][0].n + outer[1][0].n;
}

// the ordinary path, to show nothing was disturbed for a literal nobody else holds
function plainLiteral() {
    let arr = [new Leaf(1), new Leaf(2)];
    arr[0] = new Leaf(3);

    return arr[0].n + arr[1].n;
}

function main() {
    assert(twoArraysShareAValue() == 10, "a value in two literals survives both overwrites");
    assert(threeArraysShareAValue() == 7, "and survives three");
    assert(oneArrayHoldsItTwice() == 8, "a value in two slots of one literal survives both");
    assert(twoObjectLiteralsShareAValue() == 10, "the same holds for boxed object literals");
    assert(oneObjectHoldsItTwice() == 8, "and for one boxed literal holding it twice");
    assert(nestedLiterals() == 12, "an array literal captured by another survives the overwrite");
    assert(plainLiteral() == 5, "a literal nobody else holds still behaves");

    print("done.");
}
