function main1() {
    let a: number | string;

    a = "Hello";

    if (typeof (a) == "string") {
        print("str val:", a);
        assert(a == "Hello");
    }

    a = 10.0;

    if (typeof (a) == "number") {
        print("num val:", a);
        assert(a == 10.0);
    }
}

function main2() {
    let a: number | string;
    let b: number | string | boolean;

    a = 10.0;

    b = a;

    if (typeof (b) == "number") {
        print("b number: ", b);
        assert(b == 10.0);
    }
    else {
        assert(false);
    }
}

function main() {
    main1();
    main2();
    print("done.")
}