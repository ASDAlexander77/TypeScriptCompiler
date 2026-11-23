// TSLANG_ISSUES 161: `===` and `!==` between a union of primitives and a value converted the member
// it held to the value's type first, so `"1.50" === 1.5` of a `number | string` was true. A member of
// another kind is now simply not equal; `==` and `!=` still convert, as in JavaScript.
function strictEq(u: number | string, v: number) { return u === v; }
function strictNe(u: number | string, v: number) { return u !== v; }
function looseEq(u: number | string, v: number) { return u == v; }
function sameKind(u: number | string, s: string) { return u === s; }
function withBool(u: number | boolean) { return u === 1; }
function inSwitch(u: number | string) {
    switch (u) {
        case 1.5: return "number";
        case "1.50": return "string";
    }
    return "none";
}

function main() {
    assert(!strictEq("1.50", 1.5), "a string is not === a number");
    assert(strictEq(1.5, 1.5), "a number is === itself");
    assert(strictNe("1.50", 1.5), "a string is !== a number");
    assert(!strictNe(2, 2), "a number is not !== itself");
    assert(looseEq("1.50", 1.5), "== converts");
    assert(sameKind("a", "a") && !sameKind(1, "1"), "strings");
    assert(!withBool(true) && withBool(1), "a boolean is not === 1");
    assert(inSwitch(1.5) == "number" && inSwitch("1.50") == "string" && inSwitch("x") == "none", "switch");
    const xs: (number | string)[] = ["3", 3];
    let n = 0;
    for (const x of xs) if (x === 3) n++;
    assert(n == 1, "an integer literal");
    print("done.");
}
