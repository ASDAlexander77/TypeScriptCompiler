function f() {
    let a: i32 = 1;
    return atomicload(a, 7);
}

function main() {
    print("unreachable");
}
