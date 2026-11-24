// What a union may still be given to, now that a union with a member no member of the type it goes
// to takes is an error (*_union_member_to_class_error.ts): what TypeScript allows.

class Base {
    b = 1;
}

class Derived extends Base {
    d = 2;
}

class N {
    s = 3;
}

function wide(v: string | number | boolean) {
    return typeof v === "string" ? 1 : typeof v === "number" ? 2 : 3;
}

function nullableN(n: N | null | undefined) {
    return n ? n.s : 0;
}

function base(b: Base | null) {
    return b ? b.b : 0;
}

function text(s: string | null) {
    return s ? s.length : 0;
}

function anything(a: any) {
    return a ? 1 : 0;
}

function unknownValue(u: unknown) {
    return u ? 1 : 0;
}

function main() {
    // a subset union
    const sn: string | number = "a" + 1;
    assert(wide(sn) == 1, "a subset union");

    // null and undefined widening
    const n1: N | null = new N();
    assert(nullableN(n1) == 3, "N | null to N | null | undefined");
    let n2: N | undefined;
    n2 = new N();
    assert(nullableN(n2) == 3, "N | undefined to N | null | undefined");
    const n3: N | null | undefined = n1;
    assert(nullableN(n3) == 3, "a declaration widening");

    // a subclass member goes to its base
    const d: Derived | null = new Derived();
    assert(base(d) == 1, "Derived | null to Base | null");
    let b: Base | null = null;
    b = d;
    assert(b != null && b.b == 1, "assigned to its base");

    // string literals go to a string
    let lit: "ab" | "cde" = "cde";
    assert(text(lit) == 3, "a union of string literals to string | null");
    const s: string | undefined = lit;
    assert(s == "cde", "to string | undefined");

    // anything goes to any and unknown
    const sNull: string | null = "a" + 1;
    assert(anything(sNull) == 1 && unknownValue(sNull) == 1, "any and unknown");
    const nAny: any = n1;
    assert(nAny != null, "a declaration of any");

    // a string member goes to a number at run time (TypeScript may have narrowed it)
    let x: string | number = 1;
    const num: number = x;
    assert(num == 1, "a narrowed union");

    // an explicit cast is not checked
    const notN = <N | null>(<any>sNull);
    assert(notN != null, "a cast");

    print("done.");
}
