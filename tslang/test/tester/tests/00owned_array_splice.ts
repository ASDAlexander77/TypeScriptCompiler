// regression test: `splice` gives back the references held by the elements it removes.
//
// What `splice` deletes is memmoved over and then the array is realloc'd, so the references
// sitting in those slots were simply overwritten - dropped on the floor rather than released.
// §9.22 found this when it took the array-mutating ops and left it open deliberately: every
// other insertion point in this arc is in MLIRGen, where how many elements are involved is a
// compile-time matter, but here the count is a runtime value known only in the lowering. So
// this is the one release emitted from `LowerToLLVM` rather than from MLIRGen, which also puts
// it outside what the ownership verifier can see - hence this test.
//
// Measured: a loop splicing two of three boxed strings away held 16.4 MB under `-mm=rc`
// against 4.1 for the same program without the splice, with `none` at 28.7 so nothing was
// elided. It is 3.8 either way now. See docs/reference-counting-evaluation.md section 9.74.
//
// Releasing here is only sound because tslang's `splice` returns a COUNT, not the removed
// elements as JavaScript's does - so nothing outside can still be holding them by way of the
// return value. That was checked before the release was emitted, not assumed.
//
// The teeth are in the opposite direction from the leak. Releasing an element that something
// else still refers to frees live memory, so the cases below deliberately keep a second
// reference to spliced-out values and read it back after allocating hard over anything wrongly
// freed - a freed block keeps its contents until something reuses it.

class Box {
    tag: string;
    constructor(tag: string) { this.tag = tag; }
}

function main() {
    // 1. an element spliced out of one array is still held by another, and must survive
    let survivors: Box[] = [];
    for (let i = 0; i < 40; i++) {
        const keep = new Box(`kept-${i}`);
        let victim: Box[] = [];
        victim.push(new Box("doomed-a"));
        victim.push(keep);
        victim.push(new Box("doomed-b"));
        survivors.push(keep);
        victim.splice(0, 3);          // removes all three, including `keep`
        if (victim.length != 0) {
            assert(false, "splice removed the wrong number of elements");
        }
    }

    // 2. the same for strings, which are their own owning type
    let keptStrings: string[] = [];
    for (let i = 0; i < 40; i++) {
        const s = `str-${i}`;
        let arr: string[] = [];
        arr.push("head");
        arr.push(s);
        arr.push("tail");
        keptStrings.push(s);
        arr.splice(1, 2);
        if (arr.length != 1 || arr[0] != "head") {
            assert(false, "splice left the wrong remainder");
        }
    }

    // reuse anything released too early
    let churn = 0;
    for (let i = 0; i < 20000; i++) {
        let t: Box[] = [];
        t.push(new Box("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        t.push(new Box("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
        t.splice(0, 2);
        churn = churn + t.length + 1;
    }

    let bad = 0;
    for (let i = 0; i < 40; i++) {
        if (survivors[i].tag != `kept-${i}`) bad = bad + 1;
        if (keptStrings[i] != `str-${i}`) bad = bad + 1;
    }
    assert(bad == 0, "splice released an element something else still referenced");

    // 3. splice that inserts as well as removes
    let mixed = ["a", "b", "c", "d"];
    mixed.splice(1, 2, "X", "Y", "Z");
    assert(mixed.length == 5, "insert+remove length");
    assert(mixed[0] == "a", "insert+remove [0]");
    assert(mixed[1] == "X", "insert+remove [1]");
    assert(mixed[2] == "Y", "insert+remove [2]");
    assert(mixed[3] == "Z", "insert+remove [3]");
    assert(mixed[4] == "d", "insert+remove [4]");

    // 4. deleting more than is there - the release count is clamped to what exists,
    //    so this must not walk off the end
    let short = ["p", "q"];
    short.splice(1, 10);
    assert(short.length == 1, "over-long delete count");
    assert(short[0] == "p", "over-long delete kept the head");

    // 5. deleting nothing
    let untouched = ["m", "n"];
    untouched.splice(1, 0);
    assert(untouched.length == 2, "zero delete count");
    assert(untouched[1] == "n", "zero delete kept the tail");

    // 6. an element type that owns nothing must be unaffected
    let nums = [1, 2, 3, 4];
    nums.splice(0, 2);
    assert(nums.length == 2, "numeric splice length");
    assert(nums[0] == 3, "numeric splice [0]");
    assert(nums[1] == 4, "numeric splice [1]");

    assert(churn > 0, "churn");
    print("done.");
}
