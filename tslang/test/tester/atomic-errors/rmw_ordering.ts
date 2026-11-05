function f() {
    let a: i32 = 1;
    return atomicrmw(1, ReferenceOf(a), 2, 1);
}

function main() {
    print("unreachable");
}
