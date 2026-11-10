// TSLANG_ISSUES 45: `flag && a && b` widens `null | boolean | A` to `null | boolean | A | B`. That went
// through a `___cast` call, whose result -mm=own took for a value stored ("borrows the result of
// '___cast<...>'"); it is a plain cast now, the value keeping its tag. And `___cast` tests an instance
// with its class's `..instanceOf`, called through the vtable: for an exported class -mm=own took that
// call for one that may release a field read before it.

class A { }
class B { }
class C { }

function both(flag: boolean, a: A | null, b: B | null): boolean {
    if (flag && a && b) {
        return true;
    }
    return false;
}

function three(a: A | null, flag: boolean, c: C | null): boolean {
    return !!(a && flag && c);
}

// a widened union keeps what it holds
function widen(v: null | boolean | A): null | boolean | A | B {
    return v;
}

function describe(v: null | boolean | A | B): string {
    if (v === null) return "null";
    if (typeof v == "boolean") return v ? "true" : "false";
    if (v instanceof A) return "A";
    if (v instanceof B) return "B";
    return "?";
}

export class Texture { }
export class Descriptor { }

export class Pass {
    cleared = false;
    texture: Texture | null = null;
    descriptor: Descriptor | null = null;

    ready(): boolean {
        const texture = this.texture;
        const descriptor = this.descriptor;
        if (!this.cleared && texture && descriptor) {
            return true;
        }
        return false;
    }
}

function main() {
    assert(both(true, new A(), new B()), "a && b");
    assert(!both(true, new A(), null), "b is null");
    assert(!both(true, null, new B()), "a is null");
    assert(!both(false, new A(), new B()), "flag is false");

    assert(three(new A(), true, new C()), "!!(a && flag && c)");
    assert(!three(new A(), false, new C()), "!!(a && flag && c), flag is false");
    assert(!three(null, true, new C()), "!!(a && flag && c), a is null");

    assert(describe(widen(null)) == "null", "widened null");
    assert(describe(widen(true)) == "true", "widened true");
    assert(describe(widen(false)) == "false", "widened false");
    assert(describe(widen(new A())) == "A", "widened A");

    const pass = new Pass();
    assert(!pass.ready(), "no texture yet");
    pass.texture = new Texture();
    pass.descriptor = new Descriptor();
    assert(pass.ready(), "fields of exported classes tested in a method");
    pass.cleared = true;
    assert(!pass.ready(), "cleared");

    print("done.");
}
