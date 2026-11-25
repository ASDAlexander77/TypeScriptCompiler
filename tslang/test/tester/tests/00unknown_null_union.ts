// Unions with `unknown` or `Object` in them (TSLANG 193):
// - `a ? b : null` of `b: unknown` is `unknown | null`, stored as the box alone: its null was a null pointer read as a
//   box (`u === null` was false; the JIT crashed), and `u === null` compared the box's pointer
// - `any` given to `0 | unknown` (`(a ? b ?? null : null) ?? 0`): "'ts.Cast' op type ['!ts.any'] can't be stored"
// - an object literal given to `Object | string`: "'ts.Cast' op type [const_tuple] can't be stored"
// - two `unknown`s compared in a module variable's initializer, with memcmp declared (as the default library does):
//   "redefinition of symbol named 'memcmp'"
@nodrops declare function memcmp(dest: Opaque, src: Opaque, bytes: index): int;

let ma: unknown = 1;
let mb: unknown = null;
const mp = (ma ? mb ?? null : null) ?? 0;

function nullOr(a: boolean, b: unknown) {
    const u = a ? b : null;
    const r = u ?? "d";
    return (u === null ? "null " : "value ") + (u == null ? "nullish " : "set ") + typeof r;
}

function pick(a: unknown, b: unknown) {
    return (a ? b ?? null : null) ?? 0;
}

function main() {
    assert(nullOr(false, 5) == "null nullish string", "unknown | null holding null");
    assert(nullOr(true, 5) == "value set s32", "unknown | null holding a value");

    assert(typeof pick(1, 5) == "s32" && typeof pick(0, 5) == "s32" && typeof pick(1, null) == "s32", "any to 0 | unknown");
    assert(typeof mp == "s32", "?? of unknowns in a module variable");

    const x: Object | string = { x: 0 };
    const s: Object | string = "s";
    assert(typeof x != "string" && typeof s == "string", "an object literal given to Object | string");

    print("done.");
}
