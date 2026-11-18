// A spread of run-time length given to a rest parameter, `cat(...parts)`, packs a new array that holds
// a count of every element. Under rc the call holds it and gives it back after, as for `cat(a, b)`:
// read by spread-rest-pack.cmake, which finds the retain before the call and the release after it.
// Before, nothing gave the array back - the default library's Array.join leaked one per call.

function cat(...xs: string[]) {
    let s = "";
    for (const x of xs) {
        s += x;
    }

    return s;
}

function joined(parts: string[]) {
    return cat(...parts);
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        const parts = ["a" + i, "b" + i];
        const j = joined(parts);
        assert(j == "a" + i + "b" + i, "joined");
        total += j.length;
    }

    assert(total > 0, "total");
    print("done.");
}
