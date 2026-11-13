// wasm: a copy or fill whose size is known only at run time brings the memory's size up to date
// before memory.copy or memory.fill, which V8 checks against the shared memory's size as this thread
// last saw it (test-compile-wasm-no-bulk-copy-* look at the assembly). Natively it just has to run.
function main() {
    const items: number[] = [];
    for (let i = 0; i < 100; i++) items.push(i);

    // shift moves the rest down: a copy within the array
    assert(items.shift() == 0, "shift gives the first");
    assert(items[0] == 1 && items.length == 99, "shift moves the rest down");

    // splice moves the tail: a copy to a lower address, then one to a higher address
    items.splice(10, 5);
    assert(items[10] == 16 && items.length == 94, "splice closes the gap");
    items.splice(10, 0, 100, 101, 102);
    assert(items[10] == 100 && items[13] == 16 && items.length == 97, "splice opens a gap");

    // a copy of an array: a copy between two blocks
    const copy = [...items, 200];
    assert(copy.length == 98 && copy[96] == 99 && copy[97] == 200, "the spread copies");

    // strings: copies of run-time lengths
    let text = "";
    for (let i = 0; i < 50; i++) text += `${i},`;
    assert(text.length == 140, "the text has every number");

    print("done.");
}
