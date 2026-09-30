// `arr.length = n` on a grown array exposes slots nothing has written, and they read as zero. The
// default library's Set and Map grow their `int[]` buckets this way and take an unwritten bucket for
// an empty one; under every model but gc the new slots held whatever the allocator's memory held
// last, which broke the hash chains - `new Set([1, 2]).union(new Set([2, 3]))` lost elements, a run
// in two, depending on what the heap had been used for.
//
// So the heap is dirtied first: arrays of the same size are filled and dropped, and the one grown
// next is handed memory that still holds their values.

function dirty() {
    for (let k = 0; k < 8; k++) {
        let d: int[] = [];
        d.length = 64;
        for (let i = 0; i < 64; i++) {
            d[i] = 12345;
        }
    }
}

function sum(a: int[]) {
    let total = 0;
    for (let i = 0; i < a.length; i++) {
        total += a[i];
    }

    return total;
}

function grownFromEmpty() {
    dirty();
    let a: int[] = [];
    a.length = 64;
    return sum(a) == 0;
}

// Shrunk and grown again: the slots the shrink gave up come back empty, not with what they held.
function shrunkThenGrown() {
    let a: int[] = [];
    a.length = 64;
    for (let i = 0; i < 64; i++) {
        a[i] = 7;
    }

    a.length = 2;
    a.length = 64;
    return sum(a) == 14;
}

function main() {
    assert(grownFromEmpty(), "a grown array's new slots read as zero");
    assert(shrunkThenGrown(), "slots given up by a shrink read as zero when the array grows again");

    print("done.");
}
