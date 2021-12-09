type int = TypeOf<1>;
function main1() {
    print("try/catch 1");

    let t = 1;

    try {
        throw 2;
    } catch (v: int) {
        print("Hello ", v);
        t = v;
    }

    assert(2 == t);
}

function main2() {
    print("try/catch 2");

    let t = 1.0;

    try {
        throw 2.0;
    } catch (v: number) {
        print("Hello ", v);
        t = v;
    }

    assert(2.0 == t);
}

function main3() {
    print("try/catch 3");

    let t = 1;

    try {
        throw "Hello World";
    } catch (v: string) {
        print(v);
        t = 2;
    }

    // TODO: we need to have next line to force optimizator not to break code
    print(t);

    assert(2 == t);
}

function main4() {
    print("try/catch 4");

    let t = 1;

    try {
        throw <any>123;
    } catch (v: any) {
        print(<int>v);
        t = <int>v;
    }

    assert(t == 123);
}

class Error {
    i = 10;
}

function main5() {
    print("try/catch 5");

    let t = 1;

    try {
        throw <any>new Error();
    } catch (v: any) {
        print((<Error>v).i);
        t = (<Error>v).i;
    }

    assert(10 == t);
}

function main6() {
    print("try/catch 6");

    let t = 1;

    try {
        throw new Error();
    } catch (v: Error) {
        print(v.i);
        t = v.i;
    }

    assert(10 == t);
}

class AdvError extends Error { }

function main7() {
    print("try/catch 7");

    let t = 1;

    try {
        throw new AdvError();
    } catch (v: Error) {
        print(v.i);
        t = v.i;
    }

    assert(10 == t);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    main5();
    main6();
    main7();
    print("done.");
}
