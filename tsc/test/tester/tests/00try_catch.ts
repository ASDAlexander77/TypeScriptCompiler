type int = 1;
function main1() {
    print("try/catch 1");

    let t = 1;

    try {
        throw 1;
    } catch (v: int) {
        print("Hello ", v);
        v = t;
    }

    assert(v == t);
}

function main2() {
    print("try/catch 2");

    let t = 1;

    try {
        throw 2.0;
    } catch (v: number) {
        print("Hello ", v);
        v = t;
    }

    assert(v == t);
}

function main3() {
    print("try/catch 3");

    let t = 1;

    try {
        throw "Hello World";
    } catch (v: string) {
        print(v);
        v = t;
    }

    assert(v == t);
}

function main() {
    main1();
    main2();
    main3();
    print("done.");
}
