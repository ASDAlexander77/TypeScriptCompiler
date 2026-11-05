// -mm=own: a function that throws its parameter keeps it (the exception takes it), and an awaited
// call whose string result borrows its argument gives back a copy (the argument is released when
// the await's region ends).

class E {
    constructor(public m: string) {}
}

function reject<T, R>(reason: R): T {
    throw reason;
}

function rejectE(reason: E): int {
    throw reason;
}

// throws on one path only: the other releases the parameter
function maybeThrow(reason: E, really: boolean): int {
    if (really) {
        throw reason;
    }

    return 1;
}

function identity<T>(value: T): T {
    return value;
}

async function greet(name: string): Promise<string> {
    return "hello " + name;
}

function main() {
    let caught = "";
    try {
        rejectE(new E("plain"));
    } catch (e: E) {
        caught = e.m;
    }

    assert(caught == "plain", "a thrown parameter");

    try {
        reject<int, E>(new E("generic"));
    } catch (e: E) {
        caught = e.m;
    }

    assert(caught == "generic", "a thrown generic parameter");

    const named = new E("named");
    try {
        rejectE(named);
    } catch (e: E) {
        caught = e.m;
    }

    assert(caught == "named", "a named local given up to the throw");

    let r = 0;
    for (let i = 0; i < 100; i++) {
        try {
            r += maybeThrow(new E("x"), i % 2 == 0);
        } catch (e: E) {
            r += 10;
        }
    }

    assert(r == 550, "a throw on one path");

    let total = 0;
    for (let i = 0; i < 100; i++) {
        const s = await identity(greet("you"));
        total += s.length;
    }

    assert(total == 900, "an awaited borrowed string, copied");

    const t = await identity("lit");
    assert(t == "lit", "an awaited borrowed literal");

    print("done.");
}
