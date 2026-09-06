// An async function's frame is asked for with `aligned_alloc` and given back with plain `free`.
// Under `-mm=gc` both halves are rewritten to the collector's own pair and the mismatch never
// shows; every other model goes to the CRT heap, where the two have to agree. Each case here
// completes at least one frame, and the loop completes many, so a heap that has been corrupted
// has somewhere to say so.
//
// The awaited functions here return `i32`, which is what this file was written around back when
// that was the only kind of result an `await` could carry (see section 9.56 - the payload type,
// not the argument, was what decided it). They are left as they are: this file is about the frame
// allocator, and `00async_result_types.ts` is where the result types are covered.

let step = 3;

async function one() {
    return 1;
}

async function fromGlobal() {
    return step + step;
}

async function withDefault(n = 7) {
    return n;
}

async function throughAnother() {
    const inner = await fromGlobal();

    return inner + 1;
}

function awaitsOnce(): number {
    return await one();
}

function awaitsAnArrow(): number {
    const f = async () => 5;

    return await f();
}

function awaitsWithADefaultParameter(): number {
    return await withDefault();
}

function awaitsInSequence(): number {
    const a = await fromGlobal();
    step = 4;
    const b = await fromGlobal();
    step = 5;
    const c = await fromGlobal();

    return a + b + c;
}

function awaitsThroughAnother(): number {
    step = 10;

    return await throughAnother();
}

// Many frames in a row: each one is allocated and released, so a mismatched pair has every
// chance to be noticed rather than surviving to the end of a short program.
function awaitsInALoop(): number {
    let total = 0;
    step = 1;
    for (let i = 0; i < 64; i++) {
        total = total + await fromGlobal();
    }

    return total;
}

function main() {
    assert(awaitsOnce() == 1, "await once");
    assert(awaitsAnArrow() == 5, "await an async arrow");
    assert(awaitsWithADefaultParameter() == 7, "await with a default parameter");
    assert(awaitsInSequence() == 24, "await in sequence");
    assert(awaitsThroughAnother() == 21, "await through another async function");
    assert(awaitsInALoop() == 128, "await in a loop");

    print("done.");
}
