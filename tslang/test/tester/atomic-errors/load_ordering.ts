function f() {
    let a: i32 = 1;
    return atomicload(ReferenceOf(a), 5);
}

function main() {
    print("unreachable");
}
