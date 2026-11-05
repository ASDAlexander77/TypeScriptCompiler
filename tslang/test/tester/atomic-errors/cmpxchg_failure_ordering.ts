function f() {
    let a: i32 = 1;
    return cmpxchg(ReferenceOf(a), 1, 2, 7, 5);
}

function main() {
    print("unreachable");
}
