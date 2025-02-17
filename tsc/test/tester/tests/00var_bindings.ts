function testConstArray() {
    const arr = [1, 2, 3, 4, 5];

    const [a, b, c, ...rest] = arr
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(rest.length == 2);

    print(rest.length, rest[0], rest[1]);
}

function testArray() {
    let arr = []

    arr = [1, 2, 3, 4, 5];

    const [a, b, c, ...rest] = arr
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(rest.length == 2);

    print(rest.length, rest[0], rest[1]);
}

testConstArray();
testArray();

print("done.")
