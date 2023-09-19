function test1() {
    let a: [string, number] = ["asd", 1.0];
    print(a[0], a[1]);

    const b: [string, number] = ["asd", 1.0];
    print(b[0], b[1]);

    const c = ["asd", 1.0];
    print(c[0], c[1]);
}

function test2() {
    const d: [[number, string], number] = [[1.0, "asd"], 2.0];

    const v1 = d[0];
    const v2 = v1[0];
    print(v2);

    print(d[0][1]);
}

function test3() {
    let car2 = { manyCars: { a: "Saab", b: "Jeep" }, 7: "Mazda" };

    print(car2.manyCars.b); // Jeep
    print(car2[7]); // Mazda
}

type int = TypeOf<1>;

function tuple_cast() {
    let result: [value: int, done: boolean];

    let v: int | undefined;
    v = 1;

    result = [v, false];

    print(result[0], result[1]);

    assert(result[0] == 1);
    assert(result[1] == false);
}

function main() {
    test1();
    test2();
    test3();
    tuple_cast();

    print("done.");
}
