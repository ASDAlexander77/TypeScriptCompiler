// TODO: union of classes should be -> root class = class object, but first you need to create __object root class for all classes
class A {
    propA: number;
}

class B {
    propB: number;
}

function isA(p1: any): p1 is A {
    return p1 instanceof A;
}

function main() {
    // Union type
    const a = new A();
    a.propA = 10;
    let union: A | B = a;
    if (isA(union)) {
        print("this is A");
        assert(union.propA == 10);
    }
    else
    {
        assert(false);
    }

    print("done.");
}