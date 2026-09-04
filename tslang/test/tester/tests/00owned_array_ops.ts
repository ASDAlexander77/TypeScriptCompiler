// push, unshift and splice put a value into an array's data block through their own ops rather
// than through an assignment, so - exactly like the array literal in 00owned_literals.ts - none
// of them took a reference to what they inserted, while the block releases every element it
// holds when it dies.
//
// Same recipe as the literal case, and the same teeth: one value reaching two slots that are
// both later overwritten. Each overwrite releases what its slot held, the first release is
// cancelled by the unconsumed birth reference, and the second takes the value past zero and
// frees it while a local still holds it. Checked case by case against the compiler as it stood:
// push, unshift, splice-insert and the pushed-twice case each returned 3 where 10 or 8 was due,
// and the spread case 6 where 13 was - the last only after being strengthened, see its comment.
//
// This also closes the spread form of an array literal that 00owned_literals.ts left open:
// `[...xs]` is built by a synthesised `for..of` calling push, so it inherits push's fix rather
// than needing one of its own.
//
// pop and shift are deliberately not paired with a release, and that is not an omission. The
// block simply stops holding the element - the size shrinks past the slot, so the release
// routine never reaches it - and the reference the block held transfers to the returned value.
// That leaves the result carrying the same "+1 nobody has consumed" every freshly produced value
// already carries, which comes out with the rest of the slack rather than one op at a time. The
// two cases below are run-path coverage for that transfer, not counting tests.
//
// Still open: what splice *deletes* is memmoved over and its references dropped without a
// release. That leaks rather than over-releases, and it cannot be fixed at this level anyway -
// the number of elements to release is only known inside the lowering.
//
// See docs/reference-counting-evaluation.md section 9.22.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

// two arrays push the same value, then both overwrite it
function pushSharedValue() {
    let kept = new Leaf(7);
    let a: Leaf[] = [];
    let b: Leaf[] = [];
    a.push(kept);
    b.push(kept);
    a[0] = new Leaf(1);
    b[0] = new Leaf(2);

    return kept.n + a[0].n + b[0].n;
}

// the same through unshift
function unshiftSharedValue() {
    let kept = new Leaf(7);
    let a: Leaf[] = [];
    let b: Leaf[] = [];
    a.unshift(kept);
    b.unshift(kept);
    a[0] = new Leaf(1);
    b[0] = new Leaf(2);

    return kept.n + a[0].n + b[0].n;
}

// and through the insert half of splice
function spliceInsertSharedValue() {
    let kept = new Leaf(7);
    let a = [new Leaf(0)];
    let b = [new Leaf(0)];
    a.splice(0, 0, kept);
    b.splice(0, 0, kept);
    a[0] = new Leaf(1);
    b[0] = new Leaf(2);

    return kept.n + a[0].n + b[0].n;
}

// the spread form of a literal, which builds itself out of push
//
// This one needs three overwrites rather than two, and finding that out is the reason each case
// here was run against the unfixed compiler separately instead of trusting the file as a whole.
// The source array is itself a literal, so it already retains under the previous slice, and that
// extra reference absorbs the second release on its own - with only `a` and `b` overwritten this
// case passed either way and would have been quietly worthless. Overwriting the source too spends
// the reference the literal legitimately holds, which puts the two spread copies back on the hook
// for theirs.
function spreadLiteralSharesValue() {
    let kept = new Leaf(7);
    let src = [kept];
    let a = [...src];
    let b = [...src];
    src[0] = new Leaf(3);
    a[0] = new Leaf(1);
    b[0] = new Leaf(2);

    return kept.n + a[0].n + b[0].n + src[0].n;
}

// one array pushed twice with the same value, then both slots overwritten
function pushedTwiceIntoOneArray() {
    let kept = new Leaf(5);
    let arr: Leaf[] = [];
    arr.push(kept);
    arr.push(kept);
    arr[0] = new Leaf(1);
    arr[1] = new Leaf(2);

    return kept.n + arr[0].n + arr[1].n;
}

// the block hands its reference to the caller rather than releasing it
function popTransfersToCaller() {
    let arr = [new Leaf(3), new Leaf(4)];
    let last = arr.pop();

    return last.n + arr[0].n;
}

function shiftTransfersToCaller() {
    let arr = [new Leaf(3), new Leaf(4)];
    let first = arr.shift();

    return first.n + arr[0].n;
}

// a single overwrite after a push, which the slack still covers either way
function pushThenOverwriteOnce() {
    let arr: Leaf[] = [];
    arr.push(new Leaf(5));
    arr[0] = new Leaf(6);

    return arr[0].n;
}

function main() {
    assert(pushSharedValue() == 10, "a pushed value in two arrays survives both overwrites");
    assert(unshiftSharedValue() == 10, "the same holds for unshift");
    assert(spliceInsertSharedValue() == 10, "and for the insert half of splice");
    assert(spreadLiteralSharesValue() == 13, "a spread literal inherits push's retain");
    assert(pushedTwiceIntoOneArray() == 8, "one array holding it twice survives both overwrites");
    assert(popTransfersToCaller() == 7, "pop hands its reference to the caller");
    assert(shiftTransfersToCaller() == 7, "shift hands its reference to the caller");
    assert(pushThenOverwriteOnce() == 6, "a single overwrite after a push still behaves");

    print("done.");
}
