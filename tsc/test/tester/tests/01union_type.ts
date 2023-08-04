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

function main3()
{
    let callbackfn: ((value: number) => void) | ((value: number, index?: number) => void);
    
    callbackfn = (x:number) => { print("1 param: ",  x); assert(x === 10); };

    let cb1: (value: number) => void = callbackfn;
    cb1(10);

    callbackfn = (x:number, y?: number) => { print("2 params: ", x, y); assert(x == 20); assert (y == 1 || y == 2); };

    let cb2: (value: number, index?: number) => void = callbackfn;
    cb2(20, 1);
    cb2(20, 2);
}

type NeverIsRemoved = string | never | number;

function main4() {
    let a: NeverIsRemoved = "asd";
    print(a);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    print("done.")
}