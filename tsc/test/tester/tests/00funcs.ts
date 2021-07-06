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

type GreetFunction = (a: string) => void;
function greeter(fn: GreetFunction) {
    fn("hello");
}

function hello(a: string) {
    print(a);
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

    // nest func
    const a = addSquares(2, 3); // returns 13
    const b = addSquares(3, 4); // returns 25
    const c = addSquares(4, 5); // returns 41
    print(a, b, c);

    // nested
    print(outside()(10));

    test_func_in_objectliteral();

    greeter(hello);

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

function factorial(n: number) {
    if (n === 0 || n === 1) return 1;
    else return n * factorial(n - 1);
}

function addSquares(a: number, b: number) {
    function square(x: number) {
        return x * x;
    }
    return square(a) + square(b);
}

function outside() {
    function inside(x: number) {
        return x * 2;
    }
    return inside;
}

function test_func_in_objectliteral() {
    const createPet = function () {
        return {
            setName: function (newName: string) {
                print(newName);
            },
        };
    };

    const pet = createPet();
    pet.setName("Oliver");
}
