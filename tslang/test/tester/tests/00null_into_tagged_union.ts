// A class, string or array value holding null that is widened to a union with a null member of its
// own becomes that null member. It was tagged as its type instead - a class holding a null pointer -
// so `=== null` missed it, `typeof` said "class", and a truth test read its vtable through the null.

class A {
    a = 1;
}

class B {
    b = 2;
}

function widenClass(h: A | null): boolean | A | B | null {
    return h;
}

function widenString(s: string | null): string | number | null {
    return s;
}

function widenArray(v: number[] | null): number[] | boolean | null {
    return v;
}

function classes() {
    const n = widenClass(null);
    assert(n === null, "null A === null");
    assert(!(n !== null), "null A !== null");
    assert(typeof n == "null", "typeof null A");
    assert(!n, "null A is falsy");
    assert((n ?? true) === true, "null A ?? true");

    const v = widenClass(new A());
    assert(v !== null, "A !== null");
    assert(typeof v == "class", "typeof A");
    assert(!!v, "A is truthy");
}

function strings() {
    const n = widenString(null);
    assert(n === null, "null string === null");
    assert(typeof n == "null", "typeof null string");

    const s = widenString("x");
    assert(s !== null, "string !== null");
    assert(typeof s == "string", "typeof string");
}

function arrays() {
    const n = widenArray(null);
    assert(n === null, "null array === null");

    const v = widenArray([1, 2]);
    assert(v !== null, "array !== null");
}

function main() {
    classes();
    strings();
    arrays();
    print("done.");
}
