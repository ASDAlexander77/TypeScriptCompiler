// `v instanceof A` of a union of interfaces asks the interface the union holds (it said false), the class it
// narrows to is the object that interface holds (it read the interface's vtable as the object), and `<IB>v` is
// the interface it holds (it crashed the compiler). Not under -mm=own: an interface narrowed to a class is
// refused there ("kept here") for a plain interface too.
interface IA { a: number; }
interface IB { b: string; }
class A implements IA { a: number = 1; }
class B implements IB { b = "x"; }

function f(v: IA | IB) {
    if (v instanceof A) return "A" + v.a;
    if (v instanceof B) return "B" + v.b;
    return "?";
}

function g(v: IA | IB | null) { return v instanceof B ? "isB" : "notB"; }
function h(v: IA | IB) { if (v instanceof A) return "A"; return (<IB>v).b; }

function main() {
    const a: IA = new A();
    const b: IB = new B();
    assert(f(a) == "A1", "A through IA");
    assert(f(b) == "Bx", "B through IB");
    assert(g(b) == "isB" && g(a) == "notB" && g(null) == "notB", "with null");
    assert(h(b) == "x", "<IB>v");
    print("done.");
}
