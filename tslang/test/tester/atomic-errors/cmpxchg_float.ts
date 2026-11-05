function f() {
    let n: number = 1;
    return cmpxchg(ReferenceOf(n), 1, 2, 7, 7);
}

function main() {
    print("unreachable");
}
