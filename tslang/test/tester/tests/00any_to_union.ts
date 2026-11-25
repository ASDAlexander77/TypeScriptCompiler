// `any` to a union of primitives (`const v: number | string = a`): "'ts.Cast' op type ['!ts.any'] can't be stored" -
// the member whose tag the box has is read, a number's from any numeric tag (a literal is boxed as an s32); and a
// class's box holding no object (`c ? new P() : null` is a P) is null, not an `instanceof` of nothing (TSLANG 192)
class P {
    n = 4;
}

function box(i: number): any {
    return i == 0 ? 5 : i == 1 ? "s" : i == 2 ? new P() : null;
}

function kind(i: number): string {
    const v: number | string | P | null = box(i);
    if (typeof v === "number") return "num" + v;
    if (typeof v === "string") return "str" + v;
    if (v === null) return "null";
    return "P" + v.n;
}

function main() {
    const a: any = 5;
    const v: number | string = a;
    assert(typeof v === "number" && v == 5, "a number");

    const b: any = "s";
    const w: number | string = b;
    assert(typeof w === "string" && w == "s", "a string");

    assert(kind(0) == "num5", "a literal's tag");
    assert(kind(1) == "strs", "a string's tag");
    assert(kind(2) == "P4", "a class");
    assert(kind(3) == "null", "a class's box holding null");

    const u: number | undefined = a;
    assert(u == 5, "a union with undefined");

    print("done.");
}
