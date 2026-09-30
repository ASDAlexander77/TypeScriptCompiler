// A value type (number, an object type) or null is a union with a tag. Tested for truth it is false
// when it holds null, else as its value is; read as its value it is that member; as text it is "null"
// or the value's text
function incrementOrMinus(n: number | null) {
    if (n) {
        return n + 1;
    }

    return -1;
}

function isSet(n: number | null) {
    return n ? true : false;
}

function keyOf(p: { k: number } | null) {
    if (p) {
        return p.k;
    }

    return -1;
}

function keyOrMinus(p: { k: number } | null) {
    if (!p) {
        return -1;
    }

    return p.k;
}

function main() {
    assert(keyOrMinus({ k: 6 }) == 6, "object after early exit: value");
    assert(keyOrMinus(null) == -1, "object after early exit: null");
    assert(incrementOrMinus(2) == 3, "number: truthy value");
    assert(incrementOrMinus(null) == -1, "number: null");
    assert(incrementOrMinus(0) == -1, "number: zero is falsy");
    assert(isSet(5), "number: set");
    assert(!isSet(null), "number: not set");

    assert(keyOf({ k: 5 }) == 5, "object: value");
    assert(keyOf(null) == -1, "object: null");

    let m: number | null = 4;
    let z: number = m!;
    assert(z == 4, "non-null assertion reads the member");

    let n: number | null = 2;
    print(n);
    assert(`${n}` == "2", "number as text");
    n = null;
    print(n);
    assert(`${n}` == "null", "null as text");

    print("done.");
}
