function f1() {
    print("Hello World!");
}

function f2(x: number) {
    print(x);
}

function f3(x = 0) {
    print(x);
}

function run(f: () => void) {
    f();
}

function main() {
    const x = f1;
    x();
    run(x);

    const x2 = f2;
    x2(1);

    const x3 = f3;
    x3(2);

    (function () {
        print("Hello World!");
    })();

    nested();
    init_method();
    arrow_func();
    arrow_func_nobody();
    test_lmb_param();

    // rec. call
    let a: number, b: number, c: number, d: number, e: number;
    a = factorial(1); // a gets the value 1
    b = factorial(2); // b gets the value 2
    c = factorial(3); // c gets the value 6
    d = factorial(4); // d gets the value 24
    e = factorial(5); // e gets the value 120

    print(a, b, c, d, e);

    print("done.");
}

function init_method() {
    let greeting = function () {
        print("Hello TypeScript!");
    };

    greeting();
}

function arrow_func() {
    let sum = (x: number, y: number): number => {
        return x + y;
    };

    sum(10, 20);
}

function arrow_func_nobody() {
    let Print = () => print("Hello TypeScript");

    Print();

    let sum = (x: number, y: number) => x + y;

    sum(3, 4);
}

function nested() {
    function _x() {
        print(1);
        return 1;
    }
    function _y() {
        print(2);
        return 2;
    }

    print(_x() || _x() == 0 ? _x() : _y());
    print(!_x() && _x() != 0 ? _x() : _y());
}

function run_f(f: (x: number, y: number) => number) {
    return f(2, 3);
}

function test_lmb_param() {
    run_f((x: number, y: number) => x + y);
}

// function must have return type or will be error as type can't be detected
function factorial(n: number) {
    if (n === 0 || n === 1) return 1;
    else return n * factorial(n - 1);
}
