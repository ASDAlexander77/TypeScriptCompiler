// Overwriting an element of a `T[]` hands ownership over, exactly the way overwriting a field
// does. A `T[]` value is { data, length }, and its release routine walks the elements of the
// data block before freeing it (buildArrayBody) - the mirror of what releaseFields does for an
// instance - so before this, `arr[i] = x` was a bare `ts.Store` that gave the data block a
// reference nobody took, and dropped the outgoing one without releasing it.
//
// Unlike 00owned_fields.ts, these DO have teeth on the counting already, and finding out why
// was the point of running the check rather than assuming the answer. The same experiment -
// swap the store to release-before-retain, the classic way to free the value you are about to
// store back - leaves every field case passing but makes selfAssignElement below read back 0
// instead of 5.
//
// The asymmetry is not about elements. An array literal stores its elements without retaining
// them (`ts.CreateArray` emits no retain), so an element seeded by a literal holds only its
// birth reference and a release-first drops it straight to zero; a field filled through the
// assignment path holds birth + field and survives a stray release. That makes the literal a
// latent over-release in its own right - the array's release routine gives up a reference the
// literal never took - currently cancelled out exactly by the unconsumed birth reference.
// See section 9.20; fixing literal construction is the next slice.
//
// Not covered here: push, unshift and splice put a value into the same data block through their
// own ops rather than through an assignment, and pop and shift take one back out. The
// taking-out half asks the same question a return does, so those belong together in a later
// slice.
//
// See docs/reference-counting-evaluation.md section 9.20.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

// the plain case: overwrite an element repeatedly, read it back
function overwriteElement() {
    let arr = [new Leaf(1), new Leaf(2)];
    arr[0] = new Leaf(3);
    arr[0] = new Leaf(4);
    return arr[0].n + arr[1].n;
}

// something else still holds the value the element is about to drop
function aliasOutlivesElement() {
    let kept = new Leaf(7);
    let arr = [kept, new Leaf(1)];
    arr[0] = new Leaf(8);

    // `kept` must still be readable: the element gave up its reference, not the last one
    return kept.n + arr[0].n;
}

// the self-assignment case retain-before-release exists for
function selfAssignElement() {
    let arr = [new Leaf(5)];
    arr[0] = arr[0];
    return arr[0].n;
}

// one leaf held by two arrays, then one of them overwrites
function sharedBetweenArrays() {
    let leaf = new Leaf(4);
    let a = [leaf];
    let b = [leaf];
    a[0] = new Leaf(9);

    return b[0].n + leaf.n + a[0].n;
}

// an element assigned from another array's element
function elementFromElement() {
    let a = [new Leaf(6)];
    let b = [new Leaf(1)];
    b[0] = a[0];
    a[0] = new Leaf(2);

    return b[0].n + a[0].n;
}

// one value reaching two slots of the same array, then the slot it came from is overwritten
function aliasedAcrossElements() {
    let holder = new Leaf(3);
    let arr = [holder, new Leaf(1)];
    arr[1] = arr[0];
    arr[0] = new Leaf(10);

    return arr[0].n + arr[1].n + holder.n;
}

// writing through an element inside a loop, where the same slot is overwritten many times
function overwriteInLoop() {
    let arr = [new Leaf(0)];
    for (let i = 1; i <= 5; i++) {
        arr[0] = new Leaf(i);
    }

    return arr[0].n;
}

function main() {
    assert(overwriteElement() == 6, "the last value stored is the one read back");
    assert(aliasOutlivesElement() == 15, "a reference kept elsewhere outlives the element's");
    assert(selfAssignElement() == 5, "self-assignment does not free the value it stores back");
    assert(sharedBetweenArrays() == 17, "one array overwriting does not disturb the other");
    assert(elementFromElement() == 8, "an element assigned from another keeps both alive");
    assert(aliasedAcrossElements() == 16, "an aliased element survives the slot it came from");
    assert(overwriteInLoop() == 5, "repeated overwriting of one slot keeps the last value");

    print("done.");
}
