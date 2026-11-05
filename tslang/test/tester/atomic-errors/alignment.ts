function f() {
    let a: i32 = 1;
    return atomicload(ReferenceOf(a), 7, "", 3);
}

function main() {
    print("unreachable");
}
