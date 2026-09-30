// `T | null | undefined` accepts null as `T | null` does: undefined makes it optional, it must not take
// the null away - for a nullable T (string, a class), whose null is only there for the type checker,
// and for a value T (number), whose null is a real member of the union
class C {
    x = 1;
}

function isNullString(p: string | null | undefined) {
    return p === null;
}

function isUndefinedString(p: string | null | undefined) {
    return p === undefined;
}

function nullableString() {
    let s: string | null | undefined = null;
    assert(s === null, "string: null is null");
    assert(s !== undefined, "string: null is not undefined");

    s = undefined;
    assert(s === undefined, "string: undefined is undefined");
    assert(s !== null, "string: undefined is not null");

    s = "a";
    assert(s === "a", "string: value");
    assert(s !== null && s !== undefined, "string: value is neither");

    let r: undefined | null | string = null;
    assert(r === null, "string, other order: null is null");

    assert(isNullString(null), "string argument: null");
    assert(!isNullString(undefined), "string argument: undefined is not null");
    assert(isUndefinedString(undefined), "string argument: undefined");
    assert(!isUndefinedString(null), "string argument: null is not undefined");
    assert(!isNullString("b"), "string argument: value");
}

function isNullClass(p: C | null | undefined) {
    return p === null;
}

function nullableClass() {
    let c: C | null | undefined = null;
    assert(c === null, "class: null is null");
    assert(c !== undefined, "class: null is not undefined");

    c = undefined;
    assert(c === undefined, "class: undefined is undefined");

    c = new C();
    assert(c !== null && c !== undefined, "class: value is neither");
    assert(c.x == 1, "class: value field");

    assert(isNullClass(null), "class argument: null");
    assert(!isNullClass(undefined), "class argument: undefined is not null");
}

function isNullNumber(p: number | null | undefined) {
    return p === null;
}

function valueNumber() {
    let n: number | null | undefined = null;
    assert(n === null, "number: null is null");
    assert(n !== undefined, "number: null is not undefined");

    n = undefined;
    assert(n === undefined, "number: undefined is undefined");

    n = 2;
    assert(n !== null && n !== undefined, "number: value is neither");

    assert(isNullNumber(null), "number argument: null");
    assert(!isNullNumber(undefined), "number argument: undefined is not null");
    assert(!isNullNumber(3), "number argument: value");
}

// an optional holding undefined equals nothing - loosely, it equals null
function emptyOptional() {
    let o: string | undefined = undefined;
    assert(o !== null, "empty optional: not strictly null");
    assert(o == null, "empty optional: loosely null");
    assert(!(o != null), "empty optional: not loosely other than null");
    assert(o !== "", "empty optional: not an empty string");

    let b: boolean | undefined = undefined;
    assert(b !== false, "empty optional: not false");
}

interface I {
    k: number;
}

class B {
    k: number = 2;
}

function keyOf(p: I | null | undefined) {
    return p ? p.k : -1;
}

function nullableKeyOf(p: I | null) {
    return p ? p.k : -1;
}

// what `T | null` takes is cast to T first: an object literal to an interface, a const array to an array
function castToNullable() {
    assert(keyOf({ k: 1 }) == 1, "object literal to optional nullable interface");
    assert(keyOf(null) == -1, "null to optional nullable interface");
    assert(keyOf(undefined) == -1, "undefined to optional nullable interface");
    assert(nullableKeyOf({ k: 3 }) == 3, "object literal to nullable interface");
    assert(nullableKeyOf(new B()) == 2, "class to nullable interface");

    let a: number[] | null = [1, 2];
    assert(a !== null, "const array to nullable array");
    a = null;
    assert(a === null, "null to nullable array");
}

function main() {
    nullableString();
    nullableClass();
    valueNumber();
    emptyOptional();
    castToNullable();
    print("done.");
}
