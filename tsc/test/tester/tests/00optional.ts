function math() {
    let v: number | undefined;
    v = 1;
    print(v);
    assert(v == 1);
    v += 2;
    print(v);
    assert(v == 3);
    v += v;
    print(v);
    assert(v == 6);
    v++;
    print(v);
    assert(v == 7);
    --v;
    print(v);
    assert(v == 6);
}

function logic() {
    let v1: number | undefined;
    v1 = 1;
    let v2: number | undefined;
    v2 = 10;

    assert(v1 < v2);
    assert(<number>v1 < <number>v2);

    assert(v1 != v2);
    assert(<number>v1 != <number>v2);

    assert(!(v1 >= v2));
    assert(!(<number>v1 >= <number>v2));

    assert(!(v1 == v2));
    assert(!(<number>v1 == <number>v2));
}

function main() {
    math();
    logic();
    print("done.");
}
