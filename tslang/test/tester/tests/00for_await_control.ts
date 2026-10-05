// `break` and `continue` in a `for await` body, and a `for await` nested in another (#502): the
// body was generated in an async.execute region of its own, where a `break` had no loop to leave,
// and the compiler crashed.
let log = "";
async function* numbers() {
    yield 1;
    yield 2;
    yield 3;
    yield 4;
}
async function* letters() {
    yield "a";
    yield "b";
}
async function note(s: string) {
    log += s;
}
async function main() {
    for await (const x of numbers()) {
        if (x == 3) break;
        log += x;
    }
    assert(log == "12", "break");

    log = "";
    for await (const x of numbers()) {
        if (x == 2) continue;
        log += x;
    }
    assert(log == "134", "continue");

    log = "";
    for await (const x of numbers()) {
        for await (const y of letters()) {
            log += x + y;
        }
    }
    assert(log == "1a1b2a2b3a3b4a4b", "nested");

    log = "";
    for await (const x of numbers()) {
        await note("<" + x);
        if (x == 2) continue;
        if (x == 4) break;
        await note(">");
    }
    assert(log == "<1><2<3><4", "awaits, continue and break");

    print("done.");
}
