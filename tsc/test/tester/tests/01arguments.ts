function defaultArgs1(x: number, y = 3, z = 7) {
    return x;
}

function defaultArgs2(x: number, y = 3, z = 7) {
    return y;
}

function defaultArgs3(x: number, y = 3, z = 7) {
    return z;
}

function testDefaultArgs() {
    assert(defaultArgs1(1) == 1, "defl0")
    assert(defaultArgs2(1, 4) == 4, "defl1")
    assert(defaultArgs3(1, 4, 8) == 8, "defl2")
}

testDefaultArgs();
