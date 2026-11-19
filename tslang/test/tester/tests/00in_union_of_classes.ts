// `"k" in x` for `x: A | B`: the classes that have k are known from their types, and which class x is,
// asked at run time when some have it and some do not
class A { a = "a"; }
class B { b = "b"; m() { return 1; } }
class C extends B { c = 1; }
class D { a = 1; }

function hasA(x: A | B) { return "a" in x; }
function hasM(x: A | B) { return "m" in x; }
function hasC(x: B | C) { return "c" in x; }
function hasB(x: B | C) { return "b" in x; }
// two of the three have it: asked of each
function hasA3(x: A | B | D) { return "a" in x; }

function main() {
    const a = new A();
    const b = new B();
    const c = new C();
    assert(hasA(a));
    assert(!hasA(b));
    assert(!hasM(a));
    assert(hasM(b));
    assert(hasC(c));
    assert(!hasC(b));
    assert(hasB(b) && hasB(c));
    const d = new D();
    assert(hasA3(a) && hasA3(d) && !hasA3(b));
    print("done.");
}
