// Overwriting a field of a class or object instance hands ownership over, the same way
// overwriting an owned local does: the incoming value gains an owner and the outgoing one loses
// one. Before this, a field store was a bare `ts.Store` - the instance's release routine
// released whatever the field held (releaseFields has always done that), but nothing ever took
// the reference it was releasing, and nothing gave up the reference an overwritten value still
// held.
//
// What these assertions currently guard is the shape and the run path, NOT the counting.
// Checked, rather than assumed: swapping the store to release-before-retain - the classic way
// to free the value you are about to store back - leaves every case below passing. It has to,
// while a freshly allocated value's birth reference is still unconsumed (step 5a's deliberate
// slack): every count sits one above the truth, so a release can never reach zero on a live
// value and nothing is ever freed early.
//
// They are written as aliasing cases anyway, and kept in every model, because that is what
// gives them teeth the moment the slack goes: each keeps its own reference to a value,
// overwrites the field that also held it, and then reads through the reference it kept. Once a
// birth reference is consumed, an over-release there frees live memory and these reads are what
// notices.
//
// See docs/reference-counting-evaluation.md section 9.19.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

class Holder {
    item: Leaf;
}

// the plain case: overwrite a field repeatedly, read it back
function overwriteField() {
    let h = new Holder();
    h.item = new Leaf(1);
    h.item = new Leaf(2);
    h.item = new Leaf(3);
    return h.item.n;
}

// something else still holds the value the field is about to drop
function aliasOutlivesField() {
    let h = new Holder();
    let kept = new Leaf(7);
    h.item = kept;
    h.item = new Leaf(8);

    // `kept` must still be readable: the field gave up its reference, not the last one
    return kept.n + h.item.n;
}

// the self-assignment case retain-before-release exists for: releasing first could drop the
// last reference and free the value about to be stored back
function selfAssign() {
    let h = new Holder();
    h.item = new Leaf(5);
    h.item = h.item;
    return h.item.n;
}

// two holders pointing at one leaf, then one of them overwrites
function sharedBetweenHolders() {
    let leaf = new Leaf(4);
    let a = new Holder();
    let b = new Holder();
    a.item = leaf;
    b.item = leaf;
    a.item = new Leaf(9);

    return b.item.n + leaf.n + a.item.n;
}

// a field assigned from another field
function fieldFromField() {
    let a = new Holder();
    let b = new Holder();
    a.item = new Leaf(6);
    b.item = a.item;
    a.item = new Leaf(2);

    return b.item.n + a.item.n;
}

function main() {
    assert(overwriteField() == 3, "the last value stored is the one read back");
    assert(aliasOutlivesField() == 15, "a reference kept elsewhere outlives the field's");
    assert(selfAssign() == 5, "self-assignment does not free the value it stores back");
    assert(sharedBetweenHolders() == 17, "one holder overwriting does not disturb the other");
    assert(fieldFromField() == 8, "a field assigned from another field keeps both alive");

    print("done.");
}
