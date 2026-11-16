// A function that returns an object literal on some paths and undefined on others: the literal of
// constants owns nothing (immortal strings), so returning it as an optional takes no reference; one
// that holds a string made at run time owns it, and the caller releases it.
function object(c: boolean) {
    if (c) return { v: "x" };
    return undefined;
}

function made(c: boolean, n: number) {
    if (c) return { v: "made" + n };
    return undefined;
}

function main() {
    for (let i = 0; i < 200; i++) {
        assert(object(true)?.v == "x", "literal");
        assert(object(false) === undefined, "undefined");
        const o = object(true);
        assert(o?.v == "x", "through a local");
        const m = made(true, i);
        assert(m?.v == "made" + i, "made");
        assert(made(false, i) === undefined, "made: undefined");
    }

    print("done.");
}
