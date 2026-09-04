// A record held inline - an object literal with no methods, a tuple - is not a heap block of its
// own. Its storage belongs to whoever holds it, and its fields are released by that holder.
//
// The fields slice excluded this case outright, reasoning that retaining into a record nothing
// releases would leak. Half of that was wrong. An owned local holding a record *does* release its
// fields: `ts.RetainSlot` and `ts.ReleaseSlot` on a record-shaped slot go through the type's own
// routines, and those walk the fields. So the local retained the field's original value and
// released whatever the field held at scope exit, while an assignment in between swapped that
// value without taking or giving anything - and two such assignments of one value released it
// twice and freed it while a local still held it.
//
// The construction half needs nothing, and that was checked rather than assumed: a literal is
// built in scratch storage nobody owns, then copied into the owned local, whose RetainSlot
// retains the fields. That balances, which is why the construction cases below pass either way
// and are here as a guard rather than as a test of this change.
//
// The rule the fix encodes is conditional, unlike the class/object one: an inline record's field
// owns exactly when the storage under it owns. A parameter's slot and the scratch storage a
// literal is built in both answer no.
//
// See docs/reference-counting-evaluation.md section 9.23.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

// the reduced case: two inline records, each assigned the same value, both released at scope exit
function twoInlineRecordsShareAValue() {
    let x = new Leaf(1);
    {
        let a = { item: new Leaf(9) };
        let b = { item: new Leaf(9) };
        a.item = x;
        b.item = x;
    }

    return x.n;
}

// the same through a record nested inside a record, which is where the rule has to recurse
function nestedInlineRecords() {
    let x = new Leaf(2);
    {
        let a = { inner: { item: new Leaf(9) } };
        let b = { inner: { item: new Leaf(9) } };
        a.inner.item = x;
        b.inner.item = x;
    }

    return x.n;
}

// Records held as elements of an array, which is where the record predicate has to defer to the
// element one on the same reference. Coverage of that path, NOT a counting test: the releases
// here would come from the array's own release routine, and that never runs while its data block
// still carries an unconsumed birth reference. It passes either way today and gains teeth with
// everything else when the slack goes.
function recordsInsideAnArray() {
    let x = new Leaf(3);
    {
        let arr = [{ item: new Leaf(9) }, { item: new Leaf(9) }];
        arr[0].item = x;
        arr[1].item = x;
    }

    return x.n;
}

// Three records rather than two, because an array is also holding the value: the literal retains
// it legitimately, so that reference has to be spent before the unretained ones can take it past
// zero. The same arithmetic the spread case in 00owned_array_ops.ts needed.
function threeRecordsWithAnArrayHolder() {
    let x = new Leaf(4);
    let arr = [x];
    {
        let a = { item: new Leaf(9) };
        let b = { item: new Leaf(9) };
        let c = { item: new Leaf(9) };
        a.item = x;
        b.item = x;
        c.item = x;
    }

    return x.n + arr[0].n;
}

// construction, which balanced already - kept as a guard on that balance
function constructionStaysBalanced() {
    let kept = new Leaf(7);
    { let a = { item: kept }; }
    { let b = { item: kept }; }
    { let c = { item: kept }; }

    return kept.n;
}

// the ordinary path: one assignment through an inline record's field
function inlineRecordSingleAssign() {
    let a = { item: new Leaf(1) };
    a.item = new Leaf(5);

    return a.item.n;
}

function main() {
    assert(twoInlineRecordsShareAValue() == 1, "a value assigned into two inline records survives both");
    assert(nestedInlineRecords() == 2, "the rule recurses through a nested record");
    assert(recordsInsideAnArray() == 3, "and through records held as array elements (coverage only)");
    assert(threeRecordsWithAnArrayHolder() == 8, "records spend their own references before the array's");
    assert(constructionStaysBalanced() == 7, "constructing a record from a value stays balanced");
    assert(inlineRecordSingleAssign() == 5, "a single assignment through a record field behaves");

    print("done.");
}
