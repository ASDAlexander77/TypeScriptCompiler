function main() {
    let [a, b] = [1, 2];

    print(a, b);

    [a, b] = [3, 4];

    print(a, b);

    t();
    t2();

    print("done.");
}

function t() {
    let a = 1;
    let b = 3;

    [a, b] = [b, a];
    print(a); // 3
    print(b); // 1

    assert(a == 3);
    assert(b == 1);
}

function f() {
    return [1, 2];
}

function t2() {
    let [a, b] = f();
    print(a); // 1
    print(b); // 2

    assert(a == 1);
    assert(b == 2);
}
