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

function testConstTuple() {
    const a = [1, 2, "asd", true];

    const [a0, ...aa] = a;
    
    print(a0, aa);
    
    assert(a0 == 1);
    assert(aa[0] == 2);
    assert(aa[1] == "asd");
    assert(aa[2]);
}

function testTuple() {
    let a = [1, 2, "asd", true];

    const [a0, ...aa] = a;
    
    print(a0, aa);
    
    assert(a0 == 1);
    assert(aa[0] == 2);
    assert(aa[1] == "asd");
    assert(aa[2]);
}

testConstArray();
testArray();
testConstTuple();
testTuple();

print("done.")
