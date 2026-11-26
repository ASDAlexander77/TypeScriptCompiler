// A rest element in an array pattern of an iterable (`const [first, ...rest] = gen()`): the rest is an
// array of what the iterator gives after the elements before it. It was refused ("spread is not
// implemented"). Runtime strings, so -mm=own's counts are checked too.
function* strings(k: number) {
    yield "g" + k;
    yield "h" + k;
    yield "i" + k;
}

function* none() {
    if (false) yield "never";
}

function restOf(k: number): number {
    const [first, ...rest] = strings(k);
    assert(first == "g" + k, "first");
    assert(rest.length == 2, "rest length");
    assert(rest[0] == "h" + k && rest[1] == "i" + k, "rest elements");
    return rest.length;
}

function main() {
    const [a, b, ...tail] = strings(1);
    assert(a == "g1" && b == "h1", "a, b");
    assert(tail.length == 1 && tail[0] == "i1", "tail");

    const [...all] = strings(2);
    assert(all.length == 3 && all[2] == "i2", "all");

    const [x, y, z, ...empty] = strings(3);
    assert(empty.length == 0, "nothing left");

    const [...nothing] = none();
    assert(nothing.length == 0, "an empty iterator");

    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        n += restOf(i);
    }

    assert(n == 40000, "many times");

    print("done.");
}
