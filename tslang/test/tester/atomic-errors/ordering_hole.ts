function f() {
    let a: i32 = 1;
    return atomicrmw(1, ReferenceOf(a), 2, 3);
}

function main() {
    print("unreachable");
}
