function main() {
    let a: [name: string, age: number];

    a.name = "Test1";

    print(a.name, " << end (should be value Test1)");

    let b: [name: string, age: number, _8: number] = ["user", 10.0, 20.0];

    print(b.name, b.age, b._8);

    let c: [user: [name: string, age: number], type: number] = [["user2", 11.0], 1.0];

    print(c.user.name, c.user.age, c.type);
    print(c.user.name);

    c.user.name = "Test2";

    assert(c.user.name == "Test2");

    print(c.user.name, " << end (should be value Test2)");

    let a: [v1: number, v2: string];
    a = [10.0 + 20.0, "asd"];

    print(a.v1, a.v2);

    assert(a.v1 == 30.0);
    assert(a.v2 == "asd");

    print("done.");
}