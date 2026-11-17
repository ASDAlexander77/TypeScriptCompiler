// `map` and `filter` of an array of `number | string` under -mm=own: the generator that reads the
// elements kept each in its state, a second owner of a string the array holds ("a value that borrows an
// element cannot be stored, returned or captured"; with the default library, "'filter' would give an
// element of type 'number | string' a second owner"). A `number | string`, as a string, is copied
// instead: its string payload when its tag says string. Strings made at run time, in a loop, so a
// double release or a leak shows; the same program under every model.
function main() {
    let lengths = 0;
    let strings = 0;
    for (let i = 0; i < 1000; i++) {
        const a: (number | string)[] = [i, "s" + i, i * 2, "t" + i];
        for (const k of a.map(x => (typeof x == "string" ? x : "n"))) {
            lengths += k.length;
        }

        for (const k of a.filter(x => typeof x == "string")) {
            strings++;
        }
    }

    // "n" twice per round, and "s<i>", "t<i>": 2 + 2 * (1 + digits of i)
    let expected = 0;
    for (let i = 0; i < 1000; i++) {
        const digits = i < 10 ? 1 : i < 100 ? 2 : 3;
        expected += 2 + 2 * (1 + digits);
    }

    assert(lengths == expected, "map");
    assert(strings == 2000, "filter");

    print("done.");
}
