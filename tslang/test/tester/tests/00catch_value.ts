// Reading the value a `catch` clause binds.
//
// Section 9.29 found this broken, described it as "reads 0 rather than 2, but only in a module
// that throws just that one type", and drew the rule that no test should read a catch value at
// all. Section 9.65 re-measured it and neither half of that description survives. It does not
// read 0: the same binary run three times reads 134, 131, 184, which is uninitialised memory.
// And it is not about how many types the module throws - it is a back-end difference, correct
// ahead of time in every case tried and garbage under the JIT. Item 5am.
//
// This file passes in BOTH tiers, and that is worth stating because the first draft of it
// asserted otherwise and was wrong. Size is what decides which regime a module lands in: six
// catch values here is enough to be correct under the JIT as well, exactly as `00try_catch.ts`
// is. `00catch_value_minimal.ts` is the same feature cut to one clause, and it fails under the
// JIT; it is registered disabled there so the build says so.
//
// So this is the coverage section 9.29 declined to write. It is worth having on its own terms:
// `00try_catch.ts` reads catch values too, but nothing pinned the shapes below - three clauses
// of one type in a row, two payload types in one function, and a value read after its clause
// has ended.

function caughtInt() {
    let t = 0;
    try { throw 2; }
    catch (v: TypeOf<1>) { t = v; }
    return t;
}

function caughtNumber() {
    let t = 0.0;
    try { throw 2.5; }
    catch (v: number) { t = v; }
    return t;
}

function caughtString() {
    let s = "";
    try { throw "payload"; }
    catch (v: string) { s = v; }
    return s;
}

// three of the same type in a row: section 9.65 read 425, -1765822016, 425 here under the JIT,
// so each one is checked rather than just the last
function threeInARow() {
    let a = 0;
    let b = 0;
    let c = 0;
    try { throw 7; } catch (v1: TypeOf<1>) { a = v1; }
    try { throw 8; } catch (v2: TypeOf<1>) { b = v2; }
    try { throw 9; } catch (v3: TypeOf<1>) { c = v3; }
    return a * 100 + b * 10 + c;
}

// two different payload types in one function, which is the shape 9.29 believed was the fix.
// Each is checked on its own: adding them would test integer-to-float promotion instead, and
// `t + u` here reads 5 rather than 5.5 - a separate question, and not one this file is about.
let twoTypesInt = 0;
let twoTypesNumber = 0.0;

function twoTypes() {
    try { throw 2; }
    catch (v: TypeOf<1>) { twoTypesInt = v; }
    try { throw 3.5; }
    catch (w: number) { twoTypesNumber = w; }
}

// the value survives being read after the clause, not just inside it
function readAfterTheClause() {
    let t = 0;
    try { throw 41; }
    catch (v: TypeOf<1>) { t = v; }
    t = t + 1;
    return t;
}

function main() {
    assert(caughtInt() == 2, "an int catch binds the thrown value");
    assert(caughtNumber() == 2.5, "a number catch binds the thrown value");
    assert(caughtString() == "payload", "a string catch binds the thrown value");
    assert(threeInARow() == 789, "three catches of one type each bind their own value");
    twoTypes();
    assert(twoTypesInt == 2, "the int catch of a two-type function binds its own value");
    assert(twoTypesNumber == 3.5, "the number catch of a two-type function binds its own value");
    assert(readAfterTheClause() == 42, "the bound value outlives the clause");

    print("done.");
}
