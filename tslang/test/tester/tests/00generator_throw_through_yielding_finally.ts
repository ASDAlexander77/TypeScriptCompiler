// What may be thrown inside a try whose finally yields, in every model: an object caught before it gets to the finally
// (by the try's own catch clause, or a try inside it), and a string or a number thrown through it.
class Failure {
    constructor(public message: string) {}
}

let log = "";

function* caughtInside(n: number) {
    try {
        yield 1;
        try {
            throw new Failure("inner" + n);
        } catch (e) {
            log += "inner;";
        }
        yield 2;
    } finally {
        yield 3;
    }
}

function* caughtByClause(n: number) {
    try {
        yield 1;
        throw new Failure("clause" + n);
    } catch (e) {
        log += "clause;";
    } finally {
        yield 2;
    }
}

function* stringThrough(n: number) {
    try {
        yield 1;
        if (n > 0) throw "s" + n;
    } finally {
        yield 2;
    }
}

function* numberThrough(n: number) {
    try {
        yield 1;
        if (n > 0) throw n;
    } finally {
        yield 2;
    }
}

function main() {
    let sum: number = 0;
    for (const v of caughtInside(1)) sum += v;
    for (const v of caughtByClause(1)) sum += v;
    assert(sum == 9 && log == "inner;clause;", "caught before the finally: " + log);

    let seen: number = 0;
    try {
        for (const v of stringThrough(1)) seen += v;
    } catch (e) {
        log += "string;";
    }
    try {
        for (const v of numberThrough(1)) seen += v;
    } catch (e) {
        log += "number;";
    }
    assert(seen == 6 && log == "inner;clause;string;number;", "through the finally: " + log);

    print("done.");
}
