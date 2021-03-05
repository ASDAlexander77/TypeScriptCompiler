function defaultArgs(x: number, y = 3, z = 7) {
    return x + y + z;
}

function main() {
    assert(defaultArgs(1) == 11, "defl0")
    assert(defaultArgs(1, 4) == 12, "defl1")
    assert(defaultArgs(1, 4, 8) == 13, "defl2")
}
