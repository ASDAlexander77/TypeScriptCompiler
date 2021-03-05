function defaultArgs(x: number, y = 3, z = 7) {
    return x + y + z;
}

function main() {
    assert(defaultArgs(1) == 11, "defl0")
    assert(defaultArgs(1, 4) == 12, "defl1")
    assert(defaultArgs(1, 4, 8) == 13, "defl2")

    assert(optstring(3) == 6, "os0")
    assert(optstring(3, "7") == 10, "os1")
    assert(optstring2(3) == 6, "os0")
    assert(optstring2(3, "7") == 10, "os1")
}

function optstring(x: number, s?: string) {
    if (s != null) {
        return parseInt(s) + x;
    }

    return x * 2;
}

function optstring2(x: number, s: string = null) {
    if (s != null) {
        return parseInt(s) + x;
    }
    
    return x * 2;
}
