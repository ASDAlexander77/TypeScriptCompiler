// A nested array pattern of an iterable (`const [[k, v]] = map`): the element read out of an iterable is
// `T | undefined` (past its end), and the inner pattern refused it ("an array pattern of a value of type
// '[string, number] | undefined' is not supported"), under every memory model. A Map is such an
// iterable. The inner pattern now reads T's value, as TypeScript types the element T.
function* pairs(i: number) {
    yield ["a" + i, 1] as [string, number];
    yield ["b" + i, 2] as [string, number];
}

function entries(i: number): string {
    const [[k0, v0], [k1, v1]] = pairs(i);
    return k0 + "=" + v0 + "," + k1 + "=" + v1;
}

function firstOnly(i: number): string {
    const [[key]] = pairs(i);
    return key;
}

function main() {
    assert(entries(1) == "a1=1,b1=2", "two entries");
    assert(firstOnly(7) == "a7", "first key");

    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        n += entries(i).length + firstOnly(i).length;
    }

    assert(n > 0, "many times");

    print("done.");
}
