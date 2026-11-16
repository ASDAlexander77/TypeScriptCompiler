// A value merged from branches that each bring what nothing owns - a literal, null, a global never given
// another value (immortal, spec 25) - owns nothing itself: keeping it takes no reference.
class Box {
    constructor(public v: number) {}
}

const shared = new Box(1);
const other = new Box(2);

function pick(first: boolean): Box {
    return first ? shared : other;
}

function folded() {
    const text = "abc";
    let v = typeof text === "string" ? 1 : "one";
    return [v];
}

function label(c: boolean) {
    let w = c ? "yes" : "no";
    return w;
}

function main() {
    for (let i = 0; i < 100; i++) {
        assert(pick(i % 2 == 0).v == (i % 2 == 0 ? 1 : 2), "pick");
        const kept = pick(true);
        assert(kept === shared, "the global itself");
    }

    assert(folded()[0] == 1, "folded");
    assert(label(true) == "yes" && label(false) == "no", "labels");
    print("done.");
}
