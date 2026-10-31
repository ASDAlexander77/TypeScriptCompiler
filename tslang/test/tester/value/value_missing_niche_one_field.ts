// value_missing_niche.ts with the reference as the @value type's only field: T's storage `{ptr}` is
// no larger than null's `ptr`, and `T | null` (no tag) was laid out as null's bare pointer, so a
// compare with null crashed tslang and passing T to a `T | null` parameter left an
// unrealized_conversion_cast (TSLANG_ISSUES 22).
class H {
    constructor(public x: number) {}

    toString() {
        return `H${this.x}`;
    }
}

@value
interface One {
    readonly h: H;
}

@value
class Wrap {
    readonly h: H;
    constructor(h: H) {
        this.h = h;
    }
}

// a @value type holding a one-field @value type
@value
interface Outer {
    readonly one: One;
}

function f(d: number): One | null {
    return d < 0 ? null : { h: new H(d) };
}

function take(w: Wrap | null): number {
    return w ? (w as Wrap).h.x : 0;
}

function outer(d: number): Outer | null {
    if (d < 0) return null;
    return { one: { h: new H(d) } };
}

let global: One | null = null;

function main() {
    assert(f(-1) == null && !(f(1) == null), "== null");
    assert(f(-1) === null && f(1) !== null, "=== null");
    assert(global == null, "a global null");
    global = f(3);
    assert(global != null && (global as One).h.x == 3, "a global set");

    const set = f(2);
    if (set) assert(set.h.x == 2); else assert(false, "set read as null");
    const none = f(-2);
    if (none) assert(false, "null read as set");

    assert(take(new Wrap(new H(5))) == 5 && take(null) == 0, "passed to a T | null parameter");
    const o: One = { h: new H(6) };
    const held: One | null = o;
    assert(held != null && held.h.x == 6);

    assert(outer(-1) == null && (outer(4) as Outer).one.h.x == 4, "nested one-field value");

    assert(`${f(-1)}` == "null" && `${f(7)}` == "{ h: H7 }", "printed");

    print("done.");
}
