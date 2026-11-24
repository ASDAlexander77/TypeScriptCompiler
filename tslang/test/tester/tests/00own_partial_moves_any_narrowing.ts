// TSLANG 180: a value moved on some ways of an `if ... else if` with no `else` (69); `typeof a === "string"` of a
// `const a` holding a call's `any` (169); calls in a try whose finally yields of functions that throw an object only
// where they catch it, or throw a string (157)
class N {
    v = 1;
    s = "n" + 1;
}

class S {
    a: N[] = [];
    b: N[] = [];
}

function spread(rounds: number) {
    const s = new S();
    for (let i = 0; i < rounds; i++) {
        const n = new N();
        if (i % 3 == 0) s.a.push(n);
        else if (i % 3 == 1) s.b.push(n);
    }

    return s.a.length * 10 + s.b.length;
}

function nested(rounds: number) {
    const s = new S();
    for (let i = 0; i < rounds; i++) {
        const n = new N();
        if (i == 0) s.a.push(n);
        else {
            if (i == 1) s.b.push(n);
        }
    }

    return s.a.length * 10 + s.b.length;
}

function getAny(i: number): any {
    return "s" + i;
}

function textLength(i: number) {
    const a = getAny(i);
    if (typeof a === "string") {
        const s: string = a;
        return s.length;
    }

    return -1;
}

class Failure {
    msg = "x" + 1;
}

function safe() {
    try {
        throw new Failure();
    } catch (e) {
        return 1;
    }

    return 0;
}

function boom() {
    throw new Failure();
}

function text() {
    throw "s" + 1;
}

function* g() {
    try {
        yield safe();
        try {
            boom();
        } catch (e) {
            print("inner");
        }
    } finally {
        yield 2;
    }
}

function* h() {
    try {
        yield 1;
        text();
    } finally {
        yield 2;
    }
}

function main() {
    assert(spread(9) == 33 && nested(3) == 11);

    let total: number = 0;
    for (let i = 0; i < 100; i++) total += textLength(i);
    assert(total == 290);

    let sum: number = 0;
    for (const v of g()) sum += v;
    assert(sum == 3);

    let caught = false;
    try {
        for (const v of h()) sum += v;
    } catch (e) {
        caught = true;
    }

    assert(caught && sum == 6);

    print("done.");
}
