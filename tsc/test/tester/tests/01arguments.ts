function defaultArgs1(x: number, y = 3, z = 7): number {
    return x;
}

function defaultArgs2(x: number, y = 3, z = 7): number {
    return y;
}

function defaultArgs3(x: number, y = 3, z = 7): number {
    return z;
}

function defaultArgs(x: number, y = 3, z = 7) {
    return x + y + z;
}

function main() {
    assert(defaultArgs1(1) == 1, "defl0");
    assert(defaultArgs2(1, 4) == 4, "defl1");
    assert(defaultArgs3(1, 4, 8) == 8, "defl2");

    assert(defaultArgs(1) == 11, "defl0")
    assert(defaultArgs(1, 4) == 12, "defl1")
    assert(defaultArgs(1, 4, 8) == 13, "defl2")
}
