type str = "";

class Cls1 {
    foo(): number {
        print("Hello");
        return 1;
    }
}

class Cls2 extends Cls1 {}

function main() {
    assert("asd" instanceof str);

    const cls1 = new Cls1();
    const cls2 = new Cls2();

    assert(cls1 instanceof Cls1);
    assert(!(cls1 instanceof Cls2));
    assert(cls2 instanceof Cls2);
    assert(cls2 instanceof Cls1);

    print("done.");
}
