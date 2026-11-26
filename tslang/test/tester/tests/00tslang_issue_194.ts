// Bugs found by the tsc tests sweep (TSLANG 194):
// - an object literal `as const` given to a union of shapes (`const c: Candidate = { mode: "a", ... } as const`): typed by
//   `const`, its `mode` was a string no member takes ("can't be stored")
type Candidate = { mode: "a"; output: number } | { mode: "b" };

function modeOf(c: Candidate) {
    return c.mode === "a" ? "a" + c.output : "b";
}

// - a field `msg?: undefined` (a field of the type `undefined`) left out (`{ valid: true }` for `{ valid: boolean; msg?:
//   undefined }`): "field "msg" can't be found"
// - a union of shapes (a `?:` of two literals) given to another union of shapes: typeof says "tuple" for each ("Cast
//   from ... is not supported"); the tag's layout tells them apart
// - a function returning a union and then another value: its return type was a union in a union, or none at all
type Validated = { valid: boolean; msg?: undefined } | { valid: boolean; msg: string };

function plain(): { valid: boolean; msg?: undefined } {
    return { valid: false };
}

function utilValidate(b: boolean): Validated {
    return b ? { valid: true } : { valid: false, msg: "m" };
}

function validate(b: boolean) {
    if (b) {
        return utilValidate(false);
    }

    return { valid: true, msg: "x" };
}

function main() {
    const c1: Candidate = { mode: "a", output: 1 } as const;
    const c2: Candidate = { mode: "b" } as const;
    assert(modeOf(c1) == "a1" && modeOf(c2) == "b", "as const given to a union of shapes");

    assert(!plain().valid && plain().msg === undefined, "an absent field of the type undefined");
    const ok = utilValidate(true);
    const bad = utilValidate(false);
    assert(ok.valid && ok.msg === undefined && !bad.valid && bad.msg == "m", "a union of shapes given to another");
    assert(validate(true).msg == "m" && validate(false).msg == "x", "a union returned, then a member of it");

    print("done.");
}
