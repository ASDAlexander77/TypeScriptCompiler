// `let x: any;` with no initializer is undefined, as in TypeScript, in every memory model (user ruling 11-27). Outside
// -mm=rc its slot was never written: a read took whatever was on the stack, and `x === null` crashed at --opt_level=0.

function plain() {
    let u: any;
    assert(u === undefined, "=== undefined");
    assert(!(u === null), "!== null");
    assert(u == null, "== null");
    assert(u == undefined, "== undefined");
    assert(typeof u == "undefined", "typeof");
}

function assignedLater(n: number): string {
    let u: any;
    if (n > 0) {
        u = "v" + n;
    }

    return u === undefined ? "none" : <string>u;
}

// each turn of a loop declares it again, undefined
function perTurn(): number {
    let fresh = 0;
    for (let i = 0; i < 4; i++) {
        let u: any;
        if (u === undefined) {
            fresh++;
        }

        u = "t" + i;
        assert(!(u === undefined), "assigned");
    }

    return fresh;
}

function inTry(fail: boolean): string {
    try {
        let u: any;
        if (fail) {
            throw "x";
        }

        return u === undefined ? "u" : "d";
    } catch (e) {
        return "c";
    }
}

function* gen() {
    for (let i = 0; i < 2; i++) {
        let u: any;
        yield u === undefined ? 1 : 0;
        u = "g" + i;
        yield u === undefined ? 1 : 0;
    }
}

function main() {
    plain();

    assert(assignedLater(0) == "none", "never assigned");
    assert(assignedLater(3) == "v3", "assigned");
    assert(perTurn() == 4, "undefined on every turn");
    assert(inTry(false) == "u", "in a try");
    assert(inTry(true) == "c", "thrown");

    let ones = 0;
    for (const v of gen()) {
        ones += v;
    }

    assert(ones == 2, "a generator's local");

    print("done.");
}
