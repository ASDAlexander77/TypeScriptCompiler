// A class or a string holds null in its own pointer, so a union of such types with null
// (`A | B | null`, `string | A | null`) stores no null member: its value is null when its pointer is
// null, whichever member's tag it carries. The type keeps `| null`, which is what allows assigning
// null under strict null checks. A null `A` widened to `A | B | null` used to crash `=== null`, a truth
// test and `??` (the generated casts read its vtable through the null pointer), and printing such a
// union did not compile.

class A {
    a = 1;
    toString() {
        return "A";
    }
}

class B {
    b = 2;
    toString() {
        return "B";
    }
}

function widen(h: A | null): A | B | null {
    return h;
}

function literal(): A | B | null {
    return null;
}

function isNull(u: A | B | null) {
    return u === null;
}

function notNull(u: A | B | null) {
    return u != null;
}

function truthy(u: A | B | null) {
    return u ? true : false;
}

function orB(u: A | B | null): A | B {
    return u ?? new B();
}

function which(u: A | B | null) {
    if (u !== null) {
        return u instanceof A ? "A" : "B";
    }

    return "none";
}

function text(u: A | B | null): string {
    return `${u}`;
}

function nulls() {
    // a null that came in as a null A, and a literal null
    for (const u of [widen(null), literal()]) {
        assert(isNull(u), "=== null");
        assert(!notNull(u), "!= null");
        assert(!truthy(u), "falsy");
        assert(orB(u) instanceof B, "?? B");
        assert(which(u) == "none", "narrowed by !== null");
        assert(text(u) == "null", "as text");
    }
}

function values() {
    const a = widen(new A());
    assert(!isNull(a), "A === null");
    assert(notNull(a), "A != null");
    assert(truthy(a), "A truthy");
    assert(orB(a) instanceof A, "A ?? B");
    assert(which(a) == "A", "A narrowed");
    assert(text(a) == "A", "A as text");

    assert(which(new B()) == "B", "B narrowed");
}

function strings() {
    let s: string | A | null = null;
    assert(s === null, "string | A | null holding null");
    assert(!s, "null falsy");
    s = "";
    assert(s !== null, "empty string is not null");
    assert(!s, "empty string falsy");
    s = "x";
    assert(!!s, "string truthy");
    s = new A();
    assert(!!s, "A truthy in string | A | null");
}

function main() {
    nulls();
    values();
    strings();
    print("done.");
}
