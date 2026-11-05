function f() {
    let a: i32 = 1;
    return atomicrmw(99, ReferenceOf(a), 2, 7);
}

function main() {
    print("unreachable");
}
