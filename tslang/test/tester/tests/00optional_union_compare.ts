// An optional of a union compared with a value: `x == 1.5` of an `x?: number | string` (what a dictionary of
// `number | string` gives for a key). Under -mm=own it was "a value is merged from several branches": the helper
// that compares member by member narrowed the optional itself, and its string member compared with a number
// made, for an empty one, a text beside the one converted. The empty case is tested first now, and the union it
// holds compared as any union is (TSLANG 161).
function isOneAndHalf(x?: number | string) {
    return x == 1.5;
}

function isNotA1(x?: number | string) {
    return x != "a1";
}

function strictly(x?: number | string) {
    return 1.5 === x;
}

class Holder {
    v?: number | string;
}

function main() {
    for (let i = 0; i < 100; i++) {
        assert(isOneAndHalf(1.5) && !isOneAndHalf("a" + i) && !isOneAndHalf(2) && !isOneAndHalf());
        assert(!isNotA1("a" + 1) && isNotA1("a" + (i + 2)) && isNotA1(2) && isNotA1());
        assert(strictly(1.5) && !strictly("s" + i) && !strictly());

        const h = new Holder();
        assert(!(h.v == 1.5) && h.v != "x");
        h.v = "x" + i;
        assert(h.v == "x" + i && !(h.v == 1.5));
        h.v = 1.5;
        assert(h.v == 1.5 && h.v != "x" + i);
    }

    print("done.");
}
