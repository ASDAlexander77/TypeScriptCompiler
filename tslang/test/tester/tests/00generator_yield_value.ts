// A yield's value is undefined (next() takes no argument yet). It was no value at all: `const v = yield x`
// left v as whatever its field held. Whatever comes after the yield is computed in the turn that resumed.
function* gen() {
    let count = 0;
    for (let i = 0; i < 2; i++) {
        const v = yield i;
        assert(v === undefined);
        const t = "a" + i + v;
        assert(t == "a" + i + "undefined");
        count++;
    }

    let w: any = 5;
    w = (yield 10);
    assert(w === undefined);

    yield count;
}

function main() {
    let sum = 0;
    for (const v of gen()) sum += v;
    assert(sum == 0 + 1 + 10 + 2);
    print("done.");
}
