// `never` is the type with no values. A value is narrowed to it where every member was tested for: the
// default of a switch whose cases handle every kind, the `else` after the last member. A `never` variable
// takes such a value - `const _x: never = s` is the check that a kind added later gets a case - and so does a
// `never` parameter. A function that returns `never` does not come back.
type Shape = { kind: "circle", r: number } | { kind: "square", side: number };

function area(s: Shape): number {
    switch (s.kind) {
        case "circle":
            return 3 * s.r * s.r;
        case "square":
            return s.side * s.side;
        default:
            const _exhaustive: never = s;
            return _exhaustive;
    }
}

function perimeter(s: Shape): number {
    if (s.kind === "circle") {
        return 6 * s.r;
    } else if (s.kind === "square") {
        return 4 * s.side;
    } else {
        const _exhaustive: never = s;
        return _exhaustive;
    }
}

type Dir = "up" | "down";

function sign(d: Dir): number {
    switch (d) {
        case "up":
            return 1;
        case "down":
            return -1;
        default:
            return assertNever(d);
    }
}

function assertNever(x: never): never {
    throw "unexpected value";
}

function describe(v: string | number): string {
    if (typeof v === "string") return "string";
    if (typeof v === "number") return "number";
    return assertNever(v);
}

function fail(message: string): never {
    throw message;
}

// no `return` after the call: `fail` does not come back
function positive(n: number): number {
    if (n > 0) {
        return n;
    }

    fail("not positive");
}

// a type guard is whatever it returns, even of a value that has its type already: false, the value is still there
function isPositive(n: number): n is number {
    return n > 0;
}

function guarded(n: number): number {
    if (isPositive(n)) {
        return 1;
    } else {
        return n;
    }
}

// `never` parameters of a closure and of a method, called (through other lowerings than a function's)
const assertNeverArrow = (x: never): never => {
    throw "unexpected value";
};

class Checker {
    check(x: never): never {
        throw "unexpected value";
    }
}

function signArrow(d: Dir): number {
    switch (d) {
        case "up":
            return 1;
        case "down":
            return -1;
    }

    return 0;
}

function shapeArrow(s: Shape): number {
    if (s.kind === "circle") return 1;
    if (s.kind === "square") return 2;
    return assertNeverArrow(s);
}

function signMethod(d: Dir, checker: Checker): number {
    switch (d) {
        case "up":
            return 1;
        case "down":
            return -1;
        default:
            return checker.check(d);
    }
}

// a call of a `never` function in the middle of an expression ends its block there
function orFail(v: number | null, c: boolean, s: string): string {
    const a = v ?? fail("null");
    const b = c ? s : fail("false");
    const d = s || fail("empty");
    return "" + a + b + d;
}

function nothing(): void {
    return fail("nothing");
}

// `d === "up"` narrows `d` to "up", and what is left where it is false
function signEquals(d: Dir): number {
    if (d === "up") return 1;
    if (d === "down") return -1;
    return assertNever(d);
}

function signNotEquals(d: Dir): number {
    if (d !== "up") {
        return -1;
    } else {
        return 1;
    }
}

// an element `as const` keeps its literal type: what is read out of the array is "hello", which the case handles
function asConstElement(i: number): number {
    const values = ["hello" as const];
    const v = values[i];
    switch (v) {
        case "hello":
            return 1;
        default:
            const _exhaustive: never = v;
            return 0;
    }
}

function main() {
    assert(signEquals("up") == 1 && signEquals("down") == -1, "=== literal narrows");
    assert(signNotEquals("up") == 1 && signNotEquals("down") == -1, "!== literal narrows");
    assert(asConstElement(0) == 1, "as const element");
    const pair = ["x" as const, "y" as const];
    assert(pair[0] == "x" && pair[1] == "y" && pair.length == 2, "as const elements of two literals");

    assert(guarded(5) == 1 && guarded(-3) == -3, "type guard of the same type: else keeps the value");
    assert(signArrow("up") == 1 && signArrow("down") == -1, "switch with no default");
    assert(shapeArrow({ kind: "circle", r: 1 }) == 1 && shapeArrow({ kind: "square", side: 1 }) == 2,
           "never parameter of an arrow");
    assert(signMethod("up", new Checker()) == 1 && signMethod("down", new Checker()) == -1, "never parameter of a method");
    assert(orFail(1, true, "x") == "1xx", "never-returning calls not reached in expressions");

    let failures = 0;
    try { orFail(null, true, "x"); } catch (e) { failures++; }
    try { orFail(1, false, "x"); } catch (e) { failures++; }
    try { orFail(1, true, ""); } catch (e) { failures++; }
    try { nothing(); } catch (e) { failures++; }
    try {
        try { fail("in try"); } catch (e) { fail("in catch"); }
    } catch (e) {
        failures++;
    }

    assert(failures == 5, "never-returning calls throw");

    assert(area({ kind: "circle", r: 2 }) == 12, "circle area");
    assert(area({ kind: "square", side: 3 }) == 9, "square area");
    assert(perimeter({ kind: "circle", r: 2 }) == 12, "circle perimeter");
    assert(perimeter({ kind: "square", side: 3 }) == 12, "square perimeter");
    assert(sign("up") == 1 && sign("down") == -1, "switch on literals");
    assert(describe("a") == "string" && describe(1) == "number", "typeof chain");
    assert(positive(5) == 5, "never-returning call not reached");

    let caught = false;
    try {
        positive(-1);
    } catch (e) {
        caught = true;
    }

    assert(caught, "never-returning call throws");
    print("done.");
}
