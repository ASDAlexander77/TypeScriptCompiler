// A generator's locals cannot live in its frame - the state machine has to resume - so each one
// becomes a field of a heap state object, and that object's release routine gives back every
// field that owns memory. Nothing ever took those references: the frame declines ownership of
// these locals for exactly the reason the object has it, and the store into the object was not
// treated as a field gaining a value.
//
// The other half is what a capture box holds. A read-write capture puts the address of the
// variable's cell in the box; every other capture puts a copy of the value there, and that copy
// is released by the box like any other owning field. Only the cell was ever retained, so the
// value behind a by-value capture - the source array of a generator, the string a closure reads -
// was freed when the box died, which is at the end of the function that built it.
//
// Both halves are needed together, and in this order: the local retain turns the second bug from
// a silent read of a freed block into a write of a refcount into one.
//
// See docs/reference-counting-evaluation.md section 9.50.

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let words = ["zz", "yy", "xx"];
        let joined = "z" + "z";
    }
}

// The escaping generator: `filter` is a synthesised generator, and the array it walks is captured
// by value into a box the maker owned and released on the way out.
function makeFiltered(names: string[]) {
    return names.filter(x => x.length > 1);
}

function generatorOutlivesItsMaker(): number {
    let g = makeFiltered(["ab", "c", "de"]);
    churn();

    let total = 0;
    for (const s of g) total += s.length;

    return total;
}

// The same generator consumed through a spread, which is how `00spread.ts` fails: the `for...of`
// the synthesised generator runs stores the source array into a generator local. This is the case
// with teeth, and it needs all three of `--opt --opt_level=3`, the interpolated string built
// inside the callee, and printing the result - the first because the double free is only reachable
// once the optimiser proves the two pointers equal, the other two because a block that is freed
// and never reused reads back exactly as it did before.
function sum3(x = 0, y = 0, z = 0) {
    print(`Values ${x}, ${y}, ${z}`);

    return x + y + z;
}

function spreadOfFilteredArray(): number {
    const evens = [1, 2, 3, 4, 5, 6].filter(x => x % 2 == 0);

    return sum3(...evens);
}

// A generator's own local, holding a freshly built string across a yield - so the state object is
// what carries it from one resumption to the next.
function* decorated(parts: string[]) {
    for (const p of parts) {
        let open = "<" + p;
        yield open + ">";
    }
}

function generatorLocalHeldAcrossYield(): number {
    let total = 0;
    for (const s of decorated(["a", "bc"])) total += s.length;

    return total;
}

// Not generators at all: a closure over a `const` string is a by-value capture too, and the same
// missing reference frees the string when the maker returns.
type reader = () => string;

function makeGreeter(): reader {
    const greeting = "he" + "llo";

    return () => greeting;
}

function capturedByValueOutlivesMaker(): number {
    let g = makeGreeter();
    churn();

    return g().length;
}

function main() {
    assert(generatorOutlivesItsMaker() == 4, "a generator keeps what it captured by value");

    const spread = spreadOfFilteredArray();
    print(spread);
    assert(spread == 12, "a generator local owns the array it walks");

    assert(generatorLocalHeldAcrossYield() == 7, "a generator local survives a yield");
    assert(capturedByValueOutlivesMaker() == 5, "a closure keeps what it captured by value");

    print("done.");
}
