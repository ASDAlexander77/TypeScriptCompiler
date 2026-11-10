// TSLANG_ISSUES 44: the helpers made from source for a cast (`___cast`, `___unbox`, ...) are declared in
// the root namespace, and the one made for the previous cast was erased only when the code was outside a
// namespace. In a namespace the old helper stayed, so `flag && a && b` cast `null | boolean | A | B`
// with the body made for `null | boolean | A`, and a `B` threw "Can't cast from union type".

class A { }
class B { }
class C { }

namespace NS {
    export function test(flag: boolean, a: A | null, b: B | null): boolean {
        if (flag && a && b) {
            return true;
        }
        return false;
    }

    export function test3(flag: boolean, a: A | null, b: B | null, c: C | null): boolean {
        if (flag && a && b && c) {
            return true;
        }
        return false;
    }

    export namespace Inner {
        export function test(a: A | null, flag: boolean, c: C | null): boolean {
            if (a && flag && c) {
                return true;
            }
            return false;
        }
    }

    // classes of the namespace, tested in a method, as native_ts_graphics' variable_shading_nri.ts does
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
}

function main() {
    assert(NS.test(true, new A(), new B()), "a && b of two classes in a namespace");
    assert(!NS.test(true, new A(), null), "b is null");
    assert(!NS.test(true, null, new B()), "a is null");
    assert(!NS.test(false, new A(), new B()), "flag is false");

    assert(NS.test3(true, new A(), new B(), new C()), "three classes in a namespace");
    assert(!NS.test3(true, new A(), new B(), null), "c is null");

    assert(NS.Inner.test(new A(), true, new C()), "a nested namespace");
    assert(!NS.Inner.test(new A(), false, new C()), "a nested namespace, flag is false");

    const pass = new NS.Pass();
    assert(!pass.ready(), "a method: no texture yet");
    pass.texture = new NS.Texture();
    pass.descriptor = new NS.Descriptor();
    assert(pass.ready(), "a method testing classes of the namespace");
    pass.cleared = true;
    assert(!pass.ready(), "a method: cleared");

    print("done.");
}
