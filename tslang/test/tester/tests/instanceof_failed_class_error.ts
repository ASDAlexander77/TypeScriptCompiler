// `instanceof` on a union one class of which failed: that error, not a cast helper instantiating itself until the
// stack overflows (TSLANG 192).
interface Base {
    foo: string | number;
}

class Derived1 implements Base {}

class Derived2 implements Base {
    foo: number;
}

function fn3(x: Base | Derived1) {
    if (x instanceof Derived2) {
        print("Derived2");
    }
}

function main() {
    print("done.");
}
