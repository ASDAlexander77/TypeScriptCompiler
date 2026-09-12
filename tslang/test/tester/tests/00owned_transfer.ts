// `pop` and `shift` do not release the element they give up: the size shrinks past the slot, so
// the array's release routine never reaches it again, and the reference the data block held is
// handed to whoever receives the result. That makes the result already-owned, so a receiver takes
// it over instead of adding a reference of its own.
//
// What these guard is the dangerous direction. Under-consuming a transfer only leaks, and a leak
// is not observable from inside the program - so unlike 00owned_fields.ts these cannot be given
// teeth against the bug they fix. Over-consuming is what shows: the value would lose its last
// reference and be freed while the receiver still points at it.
//
// Reading through the receiver is not enough on its own to see that, and the first version of
// this file was worthless for exactly that reason - it passed with a deliberate over-release
// injected into pop, because a freed block keeps its contents until something else claims them.
// Every case therefore calls `churn()` between the transfer and the read, which allocates enough
// same-shaped blocks to land on the freed one. That is what turns a use-after-free into a wrong
// answer instead of a lucky one.
//
// Not covered, and deliberately: an ordinary call's result. Every function retains its result on
// the way out (section 9.24), so `let y = f()` is one owner above the truth too - but knowing
// which callees do that cannot be settled at the point the call is generated. See section 9.26.
//
// See docs/reference-counting-evaluation.md section 9.26.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }
}

// Allocate over whatever has just been freed. Without this a use-after-free reads the value it
// used to hold and every assertion below passes for the wrong reason.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Leaf(999);
    }
}

// the popped value must outlive the array it came out of
function popSurvivesArrayDeath() {
    let out = new Leaf(0);
    {
        let arr = [new Leaf(1), new Leaf(2)];
        out = arr.pop();
    }

    churn();
    return out.n;
}

// the same from the front
function shiftSurvivesArrayDeath() {
    let out = new Leaf(0);
    {
        let arr = [new Leaf(3), new Leaf(4)];
        out = arr.shift();
    }

    return out.n;
}

// a transferred value stored into a field, read after the array is gone
function popIntoAField() {
    let holder = new Leaf(0);
    let out = new Leaf(0);
    {
        let arr = [new Leaf(5), new Leaf(6)];
        out = arr.pop();
        holder = out;
    }

    churn();
    return out.n + holder.n;
}

// popping every element, keeping each one
function popEverything() {
    let a = new Leaf(0);
    let b = new Leaf(0);
    {
        let arr = [new Leaf(7), new Leaf(8)];
        b = arr.pop();
        a = arr.pop();
    }

    churn();
    return a.n + b.n;
}

// the array goes on being used after a transfer, so the remaining elements are untouched
function arrayStillUsableAfterPop() {
    let arr = [new Leaf(1), new Leaf(2), new Leaf(3)];
    let last = arr.pop();
    arr[0] = new Leaf(9);
    churn();

    return last.n + arr[0].n + arr[1].n;
}

// a transferred value put straight back into another array
function popThenPush() {
    let out = new Leaf(0);
    {
        let src = [new Leaf(4), new Leaf(5)];
        let dst: Leaf[] = [];
        dst.push(src.pop());
        out = dst[0];
    }

    return out.n;
}

function main() {
    assert(popSurvivesArrayDeath() == 2, "a popped value outlives the array it came from");
    assert(shiftSurvivesArrayDeath() == 3, "and so does a shifted one");
    assert(popIntoAField() == 12, "a transferred value survives reaching a second holder");
    assert(popEverything() == 15, "popping every element keeps every value");
    assert(arrayStillUsableAfterPop() == 14, "the array is undisturbed by a transfer");
    assert(popThenPush() == 5, "a transferred value can be handed straight to another array");

    print("done.");
}
