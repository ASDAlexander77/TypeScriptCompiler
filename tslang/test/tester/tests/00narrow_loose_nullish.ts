// `c == null` / `c == undefined` (loose) is true for null and for undefined alike. Of a value that may be undefined,
// the equal side was narrowed to null: in `c == null && c === undefined` the second test was false for undefined
// (and `c === null` true), as in `if (c == null) { c === undefined ... }`. It is left as it is now; the other side is
// the type without null and undefined. Results match node; every memory model.
class C {
    constructor(public x: number) {}
}

function t1(c: C | undefined): boolean { return c == null && c === undefined; }
function t2(c: C | null | undefined): boolean { return c == null && c === undefined; }
function t3(c: C | null | undefined): boolean { return c == null && c === null; }
function t4(c: C | null | undefined): boolean { return c == undefined && c === null; }
function t5(c: number | undefined): boolean { return c == null && c === undefined; }
function t6(c: string | null | undefined): boolean { return c == null && c !== undefined; }

function a(c: C | undefined): number { if (c == null) return -1; return c.x; }
function b(c: C | null | undefined): number { return c != null ? c.x : -2; }
function d(c: C | undefined): boolean { return c == null || c.x > 0; }
function e(c: number | undefined): number { if (c != undefined) { return c + 1; } return 0; }
function f(c: string | null | undefined): number { return c == undefined ? -3 : c.length; }
function g(c: C | null | undefined): string { if (c == null) { return c === undefined ? "u" : "n"; } return "c" + c.x; }

function main() {
    let out = "";
    out += `${t1(undefined)} ${t1(new C(1))}|`;
    out += `${t2(undefined)} ${t2(null)} ${t2(new C(1))}|`;
    out += `${t3(undefined)} ${t3(null)} ${t3(new C(1))}|`;
    out += `${t4(undefined)} ${t4(null)} ${t4(new C(1))}|`;
    out += `${t5(undefined)} ${t5(1)}|`;
    out += `${t6(undefined)} ${t6(null)} ${t6("s")}|`;
    out += `${a(undefined)} ${a(new C(4))} ${b(null)} ${b(undefined)} ${b(new C(5))}|`;
    out += `${d(undefined)} ${d(new C(1))} ${d(new C(-1))}|`;
    out += `${e(undefined)} ${e(3)} ${f(null)} ${f(undefined)} ${f("abc")}|`;
    out += `${g(undefined)} ${g(null)} ${g(new C(7))}`;
    print(out);
    assert(out == "true false|true false false|false true false|false true false|true false|false true false|-1 4 -2 -2 5|true true false|0 4 -3 -3 3|u n c7", "as node");
    print("done.");
}
