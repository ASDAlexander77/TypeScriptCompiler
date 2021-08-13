function foo(arg: any) {
    if (typeof arg === "string") {
        // We know this is a string now.
        print(arg);
        assert(arg === "Hello");
    }
}

function testTypeOf() {
    foo("Hello");
    foo(1);
}

class S {
    fs() {
        print("Hello");
    }
}

class S2 extends S {
    f() {
        print("Hello S2");
    }
}

function fooClass() {
    const s: S = new S2();
    if (s instanceof S2) {
        s.f();
    }
}

function testClass() {
    fooClass();
}

function main() {
    testTypeOf();
    testClass();

    print("done.");
}
