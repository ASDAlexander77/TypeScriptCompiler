// A string built at run time is a heap allocation like any other, and until section 9.37 two of
// the plainest ways to make one handed back a value nothing owned: printing a number into a
// string, and concatenating. Neither was marked as carrying a reference for its receiver, so a
// receiver added one of its own - which is balanced, and worked - while an intermediate that no
// receiver ever took was left with none at all and leaked. `"s" + k` leaks twice over: the
// conversion's result feeds the concatenation and is then forgotten.
//
// Only a leak, so the cases here cannot fail on the bug itself. What they guard is the direction
// the fix could go wrong in: a receiver now *takes over* the reference rather than adding one,
// and taking over one that was never made would free the string while it is still held.
//
// `growAndFillStrings` is the other half of the section and does fail loudly: growing an array
// exposes slots the allocator never wrote, and storing into one releases what the slot held.
//
// See docs/reference-counting-evaluation.md section 9.37.

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

class Box {
    text: string;

    constructor(text: string) {
        this.text = text;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

// A concatenation handed to a field: the field takes the reference over, so the string has to
// outlive the block that built it.
function boxConcat(k: number): Box {
    return new Box("v" + k);
}

// The whole string is compared rather than its length: a freed buffer often keeps its length
// long after its bytes have been written over, so `.length` is a much weaker reading of it.
function concatSurvivesItsBlock() {
    let held = boxConcat(7);
    churn();

    return held.text;
}

// The conversion on its own, with no concatenation over it.
function boxNumber(k: number): Box {
    return new Box(<string>k);
}

function numberToStringSurvives() {
    let held = boxNumber(1234);
    churn();

    return held.text;
}

// One string, two owners. The local holds it while the box also does, and the shorter-lived of
// the two letting go must not take it with them.
function concatSharedWithALocal(): Box {
    let s = "ab" + "cd";
    let held = new Box(s);

    return held;
}

function concatHeldTwice() {
    let held = concatSharedWithALocal();
    churn();

    return held.text;
}

// Arrays of the same shape, grown, filled and dropped, so that the block the next growth is
// handed still holds string pointers - freed ones. Dirtying the heap generally is not enough,
// and neither is doing it once: what makes an unwritten slot fatal is holding something that
// looks like a string, and eight rounds of it is what took the case from three runs in four to
// all of them.
function scratchArray() {
    let scratch: string[] = [];
    scratch.length = 3;
    for (let i = 0; i < 3; i++) {
        scratch[i] = "z" + i;
    }
}

// Growing an array hands back slots the allocator has not written, and the store into one gives
// up what the slot held first. `result.length = this.length` and then `result[i] = ..` is the
// default library's own `Array.map`, which is where this was found.
function growAndFillStrings(): string[] {
    for (let i = 0; i < 8; i++) {
        scratchArray();
    }

    let out: string[] = [];
    out.length = 3;
    for (let i = 0; i < 3; i++) {
        out[i] = "e" + i;
    }

    return out;
}

function grownArrayHoldsItsStrings() {
    let out = growAndFillStrings();
    churn();

    return out[0] + out[1] + out[2];
}

function main() {
    assert(concatSurvivesItsBlock() == "v7", "a concatenation's receiver takes it over");
    assert(numberToStringSurvives() == "1234", "printing a number into a string yields an owned one");
    assert(concatHeldTwice() == "abcd", "two owners of one string, and the first to let go is not the last");
    assert(grownArrayHoldsItsStrings() == "e0e1e2", "a grown array's new slots hold nothing to give up");

    print("done.");
}
