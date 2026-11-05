function f() {
    let n: number = 1;
    return atomicrmw(1, ReferenceOf(n), 2, 7);
}

function main() {
    print("unreachable");
}
