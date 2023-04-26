function f3() {
    function f<X, Y>(x: X, y: Y) {
        class C {
            constructor(public x: X, public y: Y) {
            }
        }
        return new C(x, y);
    }

    let c = f(10, "hello");
    let x = c.x; // number
    let y = c.y; // string

    print(x, y);

    let c2 = f("hello2", 20);
    let x2 = c2.x; // number
    let y2 = c2.y; // string

    print(x2, y2);
}

function main() {
    f3();
    print("done.");
}