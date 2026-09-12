// A global is a root: it is the only owning slot with no matching release, because it outlives
// every scope and the value in it at exit is never given back. That is why ownership skipped it -
// there is no scope to release from - and skipping it dropped the retain as well, which is a
// different thing entirely. `g = new C()` stored the instance and then gave its reference back at
// the end of the function that built it, so the global was left pointing at freed memory.
//
// Every case builds the global in one function and reads it in another, with `churn()` between,
// because a global written and read in `main` survives the bug: nothing releases until main ends.
// `nbody.ts` is the program this was found in - `init()` builds the system, and the first method
// call that reads a field out of it writes a refcount into a freed block.
//
// See docs/reference-counting-evaluation.md section 9.49.

class Holder {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Holder(999.0);
        let words = ["zz", "yy"];
        let joined = "z" + "z";
    }
}

let instance: Holder;
let names: string[];
let text: string;

function buildInstance(): void {
    instance = new Holder(5.0);
}

function buildNames(): void {
    names = ["ab", "cd"];
}

function buildText(): void {
    text = "na" + "me";
}

function instanceOutlivesItsMaker(): number {
    buildInstance();
    churn();

    return instance.x;
}

// The two array cases pass with the retain missing, and are kept for the other direction: a
// release into a global without a matching retain frees a live array, and they are what catches
// that. (Nothing frees the array at all without the fix - the heap copy a literal array is cast
// into is not a call result, so the end-of-block release never claimed it, and it leaked rather
// than dangled.) Reading `.length` would not fail either way, since an array value is
// { data, length } and the length survives in the copy - so both go through an element.
function arrayOutlivesItsMaker(): number {
    buildNames();
    churn();

    return names[0].length + names[1].length;
}

function stringOutlivesItsMaker(): number {
    buildText();
    churn();

    return text.length;
}

// The shape `nbody` actually fails on: a field read out of the global into a local of its own.
// The local retains what it binds, which is a write into the block - so a dead global is heap
// corruption here rather than a wrong answer.
function fieldOfGlobalBoundToLocal(): number {
    buildNames();
    let local = names;
    churn();

    return local[0].length + local[1].length;
}

// Overwriting a global hands the count over like any other owning slot: the incoming value gains
// an owner and the outgoing one loses one. `g = g` is the case that says the order is right - a
// release first would drop the last reference to the value being stored back.
function reassignedGlobal(): number {
    buildInstance();
    instance = new Holder(7.0);
    instance = instance;
    churn();

    return instance.x;
}

function main() {
    assert(instanceOutlivesItsMaker() == 5.0, "a global keeps the instance stored in it");
    assert(arrayOutlivesItsMaker() == 4, "a global keeps the array stored in it");
    assert(stringOutlivesItsMaker() == 4, "a global keeps the string stored in it");
    assert(fieldOfGlobalBoundToLocal() == 4, "a local can take a reference out of a global");
    assert(reassignedGlobal() == 7.0, "overwriting a global hands the count over");

    print("done.");
}
