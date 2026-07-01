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

function testUndef()
{
    let i : number | undefined = 10;
    if (i !== undefined)
        print(i);
}

function testNull()
{
    let i : number | null = 10;
    if (i !== null)
        print(i);
}

function testAmpAmp()
{
    let a: number | undefined | null = 2;

    if (a !== undefined && a !== null) {
        print(a + 2);
    }
}

function main() {
    testTypeOf();
    testClass();
    testUndef();
    testNull();
    testAmpAmp();

    print("done.");
}
