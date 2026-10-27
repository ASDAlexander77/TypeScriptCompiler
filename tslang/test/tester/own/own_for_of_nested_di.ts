// A for...of over an element read out of the array an outer for...of walks: the inner loop's array
// is a local that borrows the element. Built with debug info, the local has a record for the
// debugger of its own, which keeps nothing; it was taken for a keep ("'.a' borrows an element and
// cannot be stored"), so DefaultLib's concat failed under own with --di only.
function total(other: string[][]) {
    let n = 0;
    for (const item of other) {
        for (const v of item) n += v.length;
    }

    return n;
}

function main() {
    const y = ["a" + 1, "bc"];
    const z = ["d" + 2];
    assert(total([y, z]) == 6, "total");
    print("done.");
}
