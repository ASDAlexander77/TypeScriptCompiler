function f() {
    let a: i32 = 1;
    atomicstore(ReferenceOf(a), 1, 4);
}

function main() {
    print("unreachable");
}
